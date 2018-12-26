/*
 * Copyright 2018-2019 Ashley Harris (Maptek Australia Pty Ltd)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SPIRV_MEMORY_ANALYSER_HPP
#define SPIRV_MEMORY_ANALYSER_HPP

// This class helps generate GLSL (and friends) shaders from a complex subset of
// SPIR-V, notable those with OpMemoryModel declaring an AddressingModel not equal to
// Logical. Hand written SPIR-V, or SPIR-V compiled from GLSL, is unlikely to
// use these features, however SPIR-V generated via LLVM from front-ends like C++
// are going to be heavy in the use of pointer arithmetic, reinterpret_cast (OpBitCast),
// and other language features that are not expressible in GLSL.
//
// This class analyses all memory usage of a shader, to help turn complex code
// coming out of a C++-style optimisation pass into GLSL.
//
// For example:
//
//   OpMemoryModel(Physical64,...)
//   ...
//   %1 = OpVariable<Array<vec3, 2 members, 12 byte stride>>()
//   %2 = OpConstant(sizeof(Tvec2))
//   %3 = (... non constant)
//   %4 = (... non constant)
//   %5 = OpAccessChain(%1, 0)
//   OpStore(%5, %3)
//   %6 = OpAccessChain(%1, 1)
//   OpStore(%6, %4)
//   %7 = OpBitCast<Ptr<vec4>>(%5)
//   %8 = OpBitCast<uint64_t>(%5)
//   %9 = OpAdd(%8, %2)
//   %10 = OpBitCast<Ptr<vec4>>(%9)
//   %11 = OpLoad(%7)
//   %12 = OpLoad(%10)
//   (Usage of %11 and %12)
//
// Where this code hand-written it'd be unlikely to be expressed this way,
// however a compiler may write similar code for a general purpose target (eg
// CPU / GPU / LLVM), and then a toolchain may convert it into valid SPIRV. We
// need to decode this such that there's no pointers, including tracking changes
// to the integer %8, and, ideally, optimising out the %1 variable.
//
// In this example, by placing the %1 array in memory, and writing the %3 and %4
// values to it, tracking the pointer maths, and then simulating the loads, we
// detect that %11 and %12 are expressible as swizzles:
//
//   vec3 _3 = ...;
//   vec3 _4 = ...;
//   _11 = vec4(_3, _4.x)
//   _12 = vec4(_3.z, _4)

#include "spirv_cross.hpp"

#include <cstdint>
#include <tuple>
#include <vector>

namespace spirv_cross
{
enum class GlobalPointerComplexity
{
	// The memory analyser wasn't needed. Everything should be simple enough that the
	// existing code should manage. (Ie success!)
	AllTrivial = 0,

	// We can map all values loaded from a pointer back to composition of access chains
	// of values written to the memory or other known values. (Ie success!). Any instruction
	// that returns false to is_instruction_trivial will need additional processing,
	// detailed below.
	CanRepresentUnmodifiedWithoutPointers,

	// We can process the code, however we have to export additional code. The
	// most common things we need to add:
	// - Arrays and allocator functions. Maybe we have Ptr<T> struct members
	//   that must be stored as integer array members, or we have memory access
	//   through pointers set by recursive OpPhi instruction paths (eg. A
	//   typical compiled and optimised C++ std::array<Struct> iteration uses
	//   the pointer as the loop variable), or we may have a recursive function
	//   that calls OpVariable.
	// - We may have a "malloc" intrinsic added as an extension function.
	//   OpVariable can't be called in a loop as of current GLSL, but this may
	//   change in the future.
	// - BitCast functions for misaligned read / writes. (Of integers -
	//   misaligned float reads are too dubious to simulate).
	// - Phi tracking, as we're writing to a pointer that's set by an OpPhi, or
	//   from pointer maths based on an OpPhi with bounded values.
	//
	// This is still success.
	CanRepresentButNeedsAdditionalGlobalCode,

	// We could represent it, but we need upper bounds for the number of objects
	// created so that we can back them with an array. Maybe we have a graph or
	// a linked list? Typed pointers will need to be stored as indices into an
	// array.
	//
	// So, failure, but resolvable with a little help.
	AllocationsNeedUpperBounds,

	// We load data from a location which we couldn't statically decode. We can
	// add hints for it if we know the value. Eg we've passed in a integer and
	// an array, cast the integer to a pointer, and loaded from it. We can only
	// generate GLSL code from this if we know for certain something useful, eg
	// that the pointer will be an integer within the array.
	//
	// So, failure, but resolvable with assistance.
	UntracablePtrLoad,

	// Not going to happen. Sorry. Too hard.
	NotImplementedOrNotPossible,
};

class MemoryAnalyser
{
public:
	MemoryAnalyser(Compiler &compiler, ParsedIR &ir)
	    : compiler(compiler)
	    , ir(ir)
	{
	}

	// We can run for 32-bit and 64-bit pointer systems.
	uint32_t ptr_width = sizeof(uint32_t);

	struct WriteInfo
	{
		// The value written to this memory:
		uint32_t source_variable_id = -1;

		// Offset from the start of source (in bytes)
		uint32_t source_data_offset_bytes = 0;

		// The label of the code block in which the write is performed.
		uint32_t write_block_label = -1;

		// The write only occurs if this block has a predecessor of label.
		uint32_t conditional_predecessor_label = -1;
	};

	struct TreeNode
	{
		// Where the memory is in our virtual layout.
		uint32_t address = -1;
		uint32_t size_in_bytes = 0;

		// The chain required to get to this node from the parent.
		// May be more than one as single-element structs or size[1] arrays can be nested.
		std::vector<uint32_t> local_access_chain;

		// If known, the variable_id holding this allocation.
		uint32_t owning_variable_id = -1;

		// Whether we have a parent (eg if we're a element in an array).
		bool has_parent = false;

		// Whether we have a child (eg if we're a struct with members).
		bool has_child = false;

		// Our type.
		SPIRType type;

		// Our child entries. (Eg subdivided memory)
		std::vector<TreeNode> children;

		// Values written to the memory
		std::vector<WriteInfo> writes;
	};

	// Information about a pointer.
	struct PointerDetails
	{
		// The type pointed to by the pointer.
		SPIRType type;

		// An address stored in this pointer
		struct Address
		{
			// The address stored in this pointer, if known.
			uint32_t exact_address = uint32_t(-1);

			// If an exact address is not known, we track the morphing of the pointer
			// back from it's origin.
			uint32_t base_variable_id = uint32_t(-1);
			uint32_t offset_id = uint32_t(-1);
			int32_t litteral_offset = 0;

			// In what block the pointer is set to this address.
			uint32_t set_in_block_label = uint32_t(-1);

			// The pointer is only set to this value when the above block is entered.
			uint32_t conditional_predecessor_label = uint32_t(-1);
		};

		// All addresses stored in this pointer
		std::vector<Address> addresses;

		// Whether the pointer can hold an unlimited number of addresses.
		// For example, c++ iterators may compile to pointer maths, which
		// results in an advancing pointer in a loop. If the loop is unknown bounds,
		// we're going to have trouble converting this to an array lookup.
		//
		// Specifying a bound may allow this to be represented as an array.
		bool unbounded_address_count = false;

		// Whether we're observed to hold a value derived from a pointer, but we
		// also observed to hold a value not derived from a pointer.
		bool holds_both_pointer_and_non_pointer = false;
	};

	// Tracks each variable which holds a pointer, or an integer casted from a pointer.
	std::unordered_map<uint32_t, PointerDetails> stack_pointer_details;

	// As above, but tracks pointers stored in memory.
	std::unordered_map<uint32_t, PointerDetails> heap_pointer_details;

	// Tracks all memory allocated or otherwise referenced by the shader.
	std::vector<TreeNode> memory_tree;

	// Returns the size of a type on this platform (as we can have pointer members of
	// structs this is memory model specific).
	uint32_t size_of(const SPIRType &Type) const;

	GlobalPointerComplexity process();

	// Hint that no more than Count Type's are ever created:
	void set_hint_type_allocation_upper_bound(SPIRType Type, uint32_t Count);

	// Hint that a pointer will only point to a known object (or subregion of an object.)
	void set_hint_ptr_bounds(uint32_t UnknownPtrId, uint32_t KnownBasePtrId, uint32_t MinimumOffset,
	                         uint32_t MaximumOffset);

	void set_hint_ptr_address(uint32_t UnknownPtrId, uint32_t KnownBasePtrId, uint32_t Offset);

	// Set a hint that a pointer will have a certain alignment when memory is accessed. By default, we
	// assume that everything is always nice and aligned to multiples of the underlying scalar size,
	// so there's no mis-aligned reads or writes.
	void set_hint_ptr_alignment(uint32_t UnknownPtrId, uint32_t Modulo);

	// We may need to know the current and previous block labels. This is needed
	// to support nasty edge cases, like OpStore(OpPhi<Ptr<T>>(...), T). These
	// edge cases come from general purpose compiler tool chains (eg. LLVM)
	// applying aggressive optimisations, and are even pretty rare when compiling
	// C++ to SPIR-V.
	//
	// The label tracking is an ivec2, containing (from_label, to_label) of the
	// current block, and needs to be updated as it leaves and enters every block.
	bool is_any_label_tracking_needed() const;

	// We need to convert allocations into arrays. This is to support unbounded allocation cases.
	bool any_heap_buffers() const;
	std::vector<std::tuple<SPIRType, std::string, uint32_t>> heap_buffers() const;

	bool any_unbounded_pointers() const;
	std::vector<uint32_t> unbounded_pointers() const;

	// Returns true if the instruction can be handled without any fancy memory analysis.
	// If this returns true, no further calls are needed regarding this instruction.
	bool is_instruction_trivial(const Instruction &Instruction) const;

	// Returns true if the instructions non-trivial implementation needs no direct
	// code associated with it, so can be skipped.
	bool is_instruction_statementless(const Instruction &Instruction) const;

	// This is the worst case, a memcopy instruction over an unknown subset of an array
	// vector, or matrix. In this case the instruction will result in a 'for' loop of
	// dynamic iteration count.
	bool is_needing_dynamic_loop(const Instruction &Instruction) const;

	// This is the second worst case for a store, and should be checked next - Eg this code:
	//
	//   %1 = OpPhi<Ptr<T>>(....)
	//   %2 = OpLoad<T>(...)
	//   Store(%1, %2)
	//
	// Needs to be written as the nasty GLSL:
	//
	//   if (...) ... = _2;
	//   elseif (...) ... = _2;
	//   else ... = _2;
	//
	// (As GLSL can't have ternary operator on the LHS of an assignment)
	//
	// (Ideally these writes could be hoisted into the relevant blocks in some cases, however this is not
	// yet implemented.)
	bool is_needing_branch_write_destinations(const Instruction &Instruction) const;

	// Test if the instruction needs multiple glsl instructions to perform per copy. False example:
	//
	//   %5 = OpVariable<Array<vec4, 2>>(%3, %4)
	//   %6 = OpBitCast<size_t>(%5)
	//   %7 = OpAdd(%6, 8)
	//   %8 = OpBitCast<Ptr<vec4>>(%7)
	//   %9 = OpLoad(%8)
	//
	//   OpLoad can be represented as either:
	//
	//     vec4 _9 = (_3.wz, _4.xy);
	//
	//   Or (if the array must exist):
	//
	//     vec4 _9 = (_5[0].wz, _5[1].xy);
	//
	// True example:
	//
	//   OpName(%1, "data")
	//   %1 = OpVariable<Tuple<float, int, ivec4>>()
	//   OpStore(%1, ...)
	//   %2 = OpAccessChain<int>(%1, 1);
	//   %3 = OpLoad<ivec3>(...)
	//   %4 = OpBitCast<Ptr<iVec3>(%2)
	//   OpStore(%4, %3)
	//
	//   OpStore needs two statements to represent it, this is the third worst case (which can hypothetically co-exist with
	//   the worst and second worst case for even more worseness):
	//
	//     data._1 = _3.x
	//     data._2.xy = _3.yz
	//
	bool is_needing_multiple_assignments(const Instruction &Instruction) const;

	// We need to emit instruction(s) with a modified destination, but that modification is
	// static and known at compile time.
	//
	//   %1 = OpVariable<Tvec3>(...)
	//   %2 = OpVariable<Tvec4>(...)
	//   OpCopyMemorySized(%2, %1, 12)
	//
	// OpCopyMemorySized becomes
	//   _2.xyz = _1
	bool is_needing_new_constexpr_destination(const Instruction &Instruction) const;

	// As above, but the source needs modifying aswell / instead.
	//
	// Eg taking a subset of a vector.
	bool is_needing_new_constexpr_source(const Instruction &Instruction) const;

	// We need to emit an instruction with a modified destination, we don't know what it is
	// at compile time, but we know there's only one destination.
	//   %5 = OpVariable<Array<float, 4>>(...)
	//   %6 = OpBitCast<size_t>(%5)
	//   %7 = OpAdd(%6, %1)                    // %1 is int, of unknown value.
	//   %8 = OpBitCast<Ptr<float>>(%7)
	//   OpStore(%8, %9)
	//
	// OpLoad becomes:
	//
	//   _5[_1 / 4] = _9;
	//
	// (As we assume by default no misaligned scalars - so %1 must be either 0, 4,
	// 8 or 12. Where this not true, then call set_hint_ptr_alignment, and you'll
	// get a multi_statement version that does bit manipulation to simulate the
	// mis-aligned write).
	bool is_needing_new_dynamic_single_destination(const Instruction &Instruction);

	struct AccessChainElement
	{
		enum AccessChainMode
		{
			// Eg "._member3" or "[2]"
			Litteral,

			// Eg [_1]
			Variable,

			// Eg .xy
			LitteralRange,

			// Eg [1 + _1 / 4]
			TransformedVariable,

			// Eg [temp] Where temp is defined as an outer loop of the write.
			RunTimeLoop
		};
		AccessChainMode mode;
		uint32_t value;
		uint32_t value_upper = 0;
		int32_t value_multiplication = 1;
		int32_t value_division = 1;
		int32_t value_add = 0;
	};

	struct Location
	{
		uint32_t root_variable;
		SPIRType root_type;
		SPIRType member_type;
		std::vector<AccessChainElement> access_chain;
	};

	struct Assignment
	{
		Location from;
		Location to;
	};

	struct Operatation
	{
		std::vector<Assignment> assignments;
	};

	struct ComplexOperatation
	{
		uint32_t current_block_label = -1;

		std::unordered_map<uint32_t, std::vector<Location>> predessor_to_destinations;
		std::unordered_map<uint32_t, std::vector<Location>> predessor_to_data;

		std::vector<std::pair<AccessChainElement, AccessChainElement>> run_time_loops;
	};

	// Returns the equiverlant statements for the very complex case when:
	// - is_needing_dynamic_loop is true (it'll give the for loop bounds)
	// - is_needing_branch_write_destinations is true (it'll give the if conditions)
	// - (or both are true - that's a nasty nested case. Fun!)
	ComplexOperatation equiverlant_multi_block_code(const Instruction &Instruction);

	// Call this if we only need a single block (see above), and is_needing_multiple_assignments
	// is true. This will give the series of statements needed for this code.
	Operatation equiverlant_code_block(const Instruction &Instruction);

	// Call this in the remaining cases, which are all single statements.
	Assignment equiverlant_statement(const Instruction &Instruction);

protected:
	Compiler &compiler;
	ParsedIR &ir;

	uint32_t type_of_temporary_id(uint32_t) const;

	// Populate the entire heap memory tree's children with details. This includes
	// identifying pointers in the heap.
	void recurse_memory_tree(TreeNode *Node);

	// Called multiple times, returns true if the potential pointer set grew from
	// detecting pointers being casted to ints, those ints modified, and then
	// casted back to pointers. Also detects when a non-pointer (and non constexpr
	// 0) value is stored in an integer previously identified as also holding a
	// pointer.
	bool find_and_track_pointers(const SPIRBlock &Block);

	// Called once per block once all pointers (and int's that are really pointers) are known,
	// different behaviour caused by Phi nodes changing values is processed here too.
	void process(const SPIRBlock &Function);

	// Decodes an access chain
	std::pair<uint32_t, uint32_t> access_chain_to_byte_offset_and_size(const SPIRType &Type, uint32_t ThisStep,
	                                                                   const uint32_t *NextStep,
	                                                                   const uint32_t *EndOfSteps);

	// What an instruction actually does to memory, (if anything.)
	std::unordered_map<uint32_t, ComplexOperatation> instruction_behavior;
};

} // namespace spirv_cross

#endif
