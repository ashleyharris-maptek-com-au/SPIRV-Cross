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
#include <map>

namespace spirv_cross
{
// Wrap "unsigned, but -1 == not defined" concept.
struct UInt32TOptional
{
	uint32_t value = uint32_t(-1);
	operator bool() const
	{
		return value != uint32_t(-1);
	}

	operator uint32_t() const
	{
		return value;
	}

	auto &operator*() const
	{
		return value;
	}

	auto &operator*()
	{
		return value;
	}

	operator uint32_t &()
	{
		return value;
	}

	UInt32TOptional() = default;
	UInt32TOptional(uint32_t in)
	    : value(in)
	{
	}
	auto &operator=(uint32_t in)
	{
		return value = in;
	}
	auto &operator=(int in)
	{
		return value = in;
	}
};

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

	struct Data
	{
		virtual ~Data()
		{
		}

		enum Resolution
		{
      // We can't resolve this value. At all.
			False,

      // Can be expressed as a single constant - this is special cased as it can be used
      // for array lengths and array lookups in glsl, and pointer arithmetic elsewhere in the
      // memory analyser.
      ConstantScalar,

      // A constant, but too complex for anything useful to a memory analyser.
      ConstantComposite,

      // Something from somewhere else in the code.
      ExistingId,

      // A glsl expression, as generated by us. Can be rhs. See glsl() to get it.
			Expression,

      // Custom statement(s), as generated by us. Must be a variable. See glsl() to get it.
			Statements,

      // A complex chain of statements, if's, loops, etc, as generated by us. See glsl() to get it.
			MultiBlock
		};

    // Information about how resolved this data is - see above.
		virtual Resolution resolved(UInt32TOptional CurrentBlock = {}, UInt32TOptional PrecedingBlock = {}) const
		{
			return False;
		}

    virtual bool needs_backing_variable() const
    {
      return false;
    }

    // The id of existing spir-v which expresses this data.
    virtual uint32_t id(UInt32TOptional CurrentBlock = {}, UInt32TOptional PrecedingBlock = {}) const
    {
      return {};
    }
	};

	struct DataComposition : Data
	{
		std::vector<std::shared_ptr<Data>> inputs;
    uint32_t output_type= 0;

		Resolution resolved(UInt32TOptional CurrentBlock, UInt32TOptional PrecedingBlock) const override
		{
			Resolution r = Expression;
			for (auto &input : inputs)
			{
				Resolution a = input->resolved(CurrentBlock, PrecedingBlock);
				if (a == False)
					return False;
				r = std::max(r, a);
			}
			return r;
		}

    bool needs_backing_variable() const override
    {
      for (auto &input : inputs)
      {
        if (input->needs_backing_variable()) return true;
      }

      return false;
    }
	};

	struct DataCast : Data
	{
		std::shared_ptr<Data> input;
    uint32_t input_type = 0;
    uint32_t output_type = 0;

		Resolution resolved(UInt32TOptional CurrentBlock, UInt32TOptional PrecedingBlock) const override
		{
			return input->resolved(CurrentBlock, PrecedingBlock);
		}

    bool needs_backing_variable() const override
    {
      if (input->needs_backing_variable()) return true;

      return false;
    }

		//std::string glsl(UInt32TOptional CurrentBlock, UInt32TOptional PrecedingBlock) const override;
	};

	struct DataBranch : Data
	{
		std::map<std::pair<uint32_t, uint32_t>, std::shared_ptr<Data>> edge_to_value;
		CFG *cfg = nullptr;

		Resolution resolved(UInt32TOptional CurrentBlock, UInt32TOptional PrecedingBlock) const override
		{
			if (PrecedingBlock && CurrentBlock)
			{
				// Look for a value set along the edge:
				auto edge = std::make_pair(*PrecedingBlock, *CurrentBlock);

				auto it = edge_to_value.find(edge);
				if (it != edge_to_value.end())
				{
					return it->second->resolved(CurrentBlock, PrecedingBlock);
				}
			}

			if (CurrentBlock)
			{
				// Look for a value set in the current block (Why didn't an optimiser pick this up?)
				auto edge = std::make_pair(uint32_t(-1), *CurrentBlock);

				auto it = edge_to_value.find(edge);
				if (it != edge_to_value.end())
				{
					return it->second->resolved(CurrentBlock, PrecedingBlock);
				}

				// Ok - try to track back through dominator’s and look for the value being set.
				auto block = *CurrentBlock;

				while (block != uint32_t(-1))
				{
					auto entries = cfg->get_preceding_edges(block);

					if (entries.size() != 1)
						break;
					block = entries.front();

					auto edge = std::make_pair(uint32_t(-1), block);
					auto it = edge_to_value.find(edge);
					if (it != edge_to_value.end())
					{
						return it->second->resolved(CurrentBlock, PrecedingBlock);
					}

					edge = std::make_pair(entries.front(), block);
					it = edge_to_value.find(edge);
					if (it != edge_to_value.end())
					{
						return it->second->resolved(CurrentBlock, PrecedingBlock);
					}
				}
			}

			if (PrecedingBlock)
			{
				// Look for a value set in the previous block (Why didn't an optimiser pick this up?)
				auto edge = std::make_pair(uint32_t(-1), *PrecedingBlock);

				auto it = edge_to_value.find(edge);
				if (it != edge_to_value.end())
				{
					return it->second->resolved(CurrentBlock, PrecedingBlock);
				}

				// Ok - try to track back through dominator’s and look for the value being set.
				auto block = *PrecedingBlock;

				while (block != uint32_t(-1))
				{
					auto entries = cfg->get_preceding_edges(block);

					if (entries.size() != 1)
						break;
					block = entries.front();

					auto edge = std::make_pair(uint32_t(-1), block);
					auto it = edge_to_value.find(edge);
					if (it != edge_to_value.end())
					{
						return it->second->resolved(CurrentBlock, PrecedingBlock);
					}

					edge = std::make_pair(entries.front(), block);
					it = edge_to_value.find(edge);
					if (it != edge_to_value.end())
					{
						return it->second->resolved(CurrentBlock, PrecedingBlock);
					}
				}
			}

			// We cna't resolve the value.
			return False;
		}

    bool needs_backing_variable() const override
    {
      for (auto &input : edge_to_value)
      {
        if (input.second->needs_backing_variable()) return true;
      }

      return false;
    }

		//std::string glsl(UInt32TOptional CurrentBlock, UInt32TOptional PrecedingBlock) const override;
	};

  struct DataConstant : Data
  {
    SPIRConstant *valueConstant = nullptr;
    SPIRConstantOp *valueConstExpr = nullptr;
    double valueActual = 0;
    uint32_t type = 0;

    bool isScalar = false;

    Resolution resolved(UInt32TOptional CurrentBlock, UInt32TOptional PrecedingBlock) const override
    {
      // We're a new constant - not an existing one.
      if (!valueConstant && !valueConstExpr) return Expression;

      if (isScalar && valueConstant)
        return ConstantScalar;

      return ConstantComposite;
    }

    uint32_t id(UInt32TOptional CurrentBlock, UInt32TOptional PrecedingBlock) const override
    {
      return valueConstExpr ? valueConstExpr->self : valueConstant->self;
    }

    bool needs_backing_variable() const override
    {
      return false;
    }
  };

  struct DataAccessChain : Data
  {
    std::shared_ptr<Data> parent_value;

    std::vector<uint32_t> access_chain;
    int32_t relative_pointer = 0;

    Resolution resolved(UInt32TOptional CurrentBlock = {}, UInt32TOptional PrecedingBlock = {}) const override
    {
      auto parent = parent_value->resolved(CurrentBlock, PrecedingBlock);

      if (parent == False)
        return False;
      if (parent == MultiBlock)
        return MultiBlock;

      //:TODO: Hunt through the IR to see if an access chain already exists - in which case we can
      // use it.

      // We need to add a custom access chain onto the end of the expression
      return Expression;
    }

    bool needs_backing_variable() const override
    {
      return parent_value->needs_backing_variable();
    }
  };

  // Value of a custom opcode, used to signal to the exporter that it needs to consult us
  // for help generating code.
  constexpr static uint16_t OpGenerateViaSpirvCrossMemoryAnalyser = 0xf191;
  constexpr static uint16_t OpMemCopyViaSpirvCrossMemoryAnalyser = 0xf192;

	struct TreeNode
	{
		// Where the memory is in our virtual layout.
		uint32_t address;
		uint32_t size_in_bytes = 0;

		// The chain required to get to this node from the parent.
		// May be more than one as single-element structs or size[1] arrays can be nested.
		std::vector<uint32_t> local_access_chain;

		// If known, the variable_id holding this allocation.
		UInt32TOptional owning_variable_id;

		// Whether we have a parent (eg if we're a element in an array).
		bool has_parent = false;

		// Whether we have a child (eg if we're a struct with members).
		bool has_child = false;

		// Our type.
		uint32_t type;

    spv::StorageClass storage_class = spv::StorageClassGeneric;

		// Our child entries. (Eg subdivided memory)
		std::vector<TreeNode> children;

		// Value written to the memory
		std::shared_ptr<Data> value;
	};

	// Information about a pointer.
	struct PointerDetails
	{
		// The type pointed to by the pointer.
		uint32_t type;

		// An address stored in this pointer
		struct Address
		{
			// The address stored in this pointer, if known.
			UInt32TOptional exact_address;

			// If an exact address is not known, we track the morphing of the pointer
			// back from it's origin.
			UInt32TOptional base_variable_id;
			UInt32TOptional offset_id;
			int32_t litteral_offset = 0;

			// In what block the pointer is set to this address.
			UInt32TOptional set_in_block_label;

			// The pointer is only set to this value when the above block is entered.
			UInt32TOptional conditional_predecessor_label;
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

		// Whether the address could ever be loaded from memory.
		bool address_ever_loaded_from_memory = false;
	};

  // Returns a value (eg the code for an OpLoad) that was previously dependant on complex
  // memory details.
  const Data* value_generated(uint32_t Id) const
  {
    auto it = id_data.find(Id);
    if (it != id_data.end()) return it->second.get();
    return nullptr;
  }

	// Tracks each variable which holds a pointer, or an integer casted from a pointer.
	std::unordered_map<uint32_t, PointerDetails> stack_pointer_details;

	// Returns every address which this pointer may be pointing at.
	// If the bounds are infinite (or practically infinite for the purpose of generating
	// branching glsl code), then an empty vector is returned.
	std::vector<uint32_t> all_potential_stack_pointer_destinations(uint32_t Id,
	                                                               UInt32TOptional LabelOfQueryingBlock = {},
	                                                               UInt32TOptional LabelOfPrecessorBlock = {}) const;

	// Returns every potential value for an integer Id (that need not be constant).
	// If the values are infinite (or practically infinite for the purpose of generating
	// branching glsl code), then an empty vector is returned.
	std::vector<uint32_t> all_potential_integer_values(uint32_t Id, UInt32TOptional LabelOfQueryingBlock = {},
	                                                   UInt32TOptional LabelOfPrecessorBlock = {}) const;

	// As above, but if no hints have been given regarding alignment and values,
	// will walk the memory tree to see what sizes we might be wanting to work
	// with based on what copy sizes would avoid undefined behaviour.
	std::vector<uint32_t> all_potential_sensible_mem_copy_sizes(uint32_t SizeId, uint32_t SourcePtrAddress,
	                                                            uint32_t DestPtrAddress,
	                                                            UInt32TOptional LabelOfQueryingBlock = {},
	                                                            UInt32TOptional LabelOfPrecessorBlock = {});

	// As above, but tracks pointers stored in memory.
	std::unordered_map<uint32_t, PointerDetails> heap_pointer_details;

	// Tracks all memory allocated or otherwise referenced by the shader.
	std::vector<TreeNode> memory_tree;

	// Returns the size of a type on this platform (as we can have pointer members of
	// structs this is memory model specific).
	uint32_t size_of(const SPIRType &Type) const;
	std::vector<uint32_t> child_types(const SPIRType &Type) const;
	bool type_holds_pointer(const SPIRType &Type) const;

  bool variable_needed(uint32_t VarId) const
  {
    return variables_to_cull.find(VarId) == variables_to_cull.end();
  }

	GlobalPointerComplexity process();

	// Hint that no more than Count Type's are ever created:
	void set_hint_type_allocation_upper_bound(SPIRType Type, uint32_t Count);

	// Hint that a pointer will only point to a known object (or subregion of an object.)
	void set_hint_ptr_bounds(uint32_t UnknownPtrId, uint32_t KnownBasePtrId, uint32_t MinimumOffset,
	                         uint32_t MaximumOffset);

	void set_hint_ptr_address(uint32_t UnknownPtrId, uint32_t KnownBasePtrId, uint32_t Offset);

	// Set a hint that a pointer will have a certain alignment when memory is accessed. By default, we
	// assume that everything is always nice and aligned to multiples of the underlying scalar size,
	// so there's no mis-aligned reads or writes. Call this to mess up this optimistic assumption.
	void set_hint_ptr_alignment(uint32_t UnknownPtrId, uint32_t Modulo);

	// Sets a hint that an integer is a known value or known range. Useful when pointer arithmatic is done with
	// a pointer and an integer which comes from outside the shader.
	void set_hint_int_bounds(uint32_t UnknownId, uint32_t Minimum, uint32_t ExclusiveMaximum, uint32_t Step = 1);
	void set_hint_int_value(uint32_t UnknownId, uint32_t Value);

	// We may need to know the current and previous block labels. This is needed
	// to support nasty edge cases, like OpStore(OpPhi<Ptr<T>>(...), T). These
	// edge cases come from general purpose compiler tool chains (eg. LLVM)
	// applying aggressive optimisations, and are even pretty rare when compiling
	// C++ to SPIR-V.
	//
	// The label tracking temp is an ivec2, containing (from_label, to_label) of
	// the current block, and needs to be updated as it leaves and enters every
	// block.
	bool is_any_label_tracking_needed() const;

	// We need to convert allocations into arrays. This is to support unbounded allocation cases.
	bool any_heap_buffers() const;
	std::vector<std::tuple<SPIRType, std::string, uint32_t>> heap_buffers() const;

	bool any_unbounded_pointers() const;
	std::vector<uint32_t> unbounded_pointers() const;

	// Returns true if the instruction can be handled without any fancy memory analysis.
	// If this returns true, no further calls are needed regarding this instruction.
	//bool is_instruction_trivial(const Instruction &Instruction) const;

	// Returns true if the instructions non-trivial implementation needs no direct
	// code associated with it, so can be skipped.
	//bool is_instruction_statementless(const Instruction &Instruction) const;

	// This is the worst case, a memcopy instruction over an unknown subset of an array
	// vector, or matrix. In this case the instruction will result in a 'for' loop of
	// dynamic iteration count.
	//bool is_needing_dynamic_loop(const Instruction &Instruction) const;

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
	//bool is_needing_branch_write_destinations(const Instruction &Instruction) const;

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
	//bool is_needing_multiple_assignments(const Instruction &Instruction) const;

	// We need to emit instruction(s) with a modified destination, but that modification is
	// static and known at compile time.
	//
	//   %1 = OpVariable<Tvec3>(...)
	//   %2 = OpVariable<Tvec4>(...)
	//   OpCopyMemorySized(%2, %1, 12)
	//
	// OpCopyMemorySized becomes
	//   _2.xyz = _1
	//bool is_needing_new_constexpr_destination(const Instruction &Instruction) const;

	// As above, but the source needs modifying aswell / instead.
	//
	// Eg taking a subset of a vector.
	//bool is_needing_new_constexpr_source(const Instruction &Instruction) const;

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
	//bool is_needing_new_dynamic_single_destination(const Instruction &Instruction);

	//struct AccessChainElement
	//{
	//	enum AccessChainMode
	//	{
	//		// Eg "._member3" or "[2]"
	//		Litteral,

	//		// Eg [_1]
	//		Variable,

	//		// Eg .xy
	//		LitteralRange,

	//		// Eg [1 + _1 / 4]
	//		TransformedVariable,

	//		// Eg [temp] Where temp is defined as an outer loop of the write.
	//		RunTimeLoop
	//	};
	//	AccessChainMode mode;
	//	uint32_t value;
	//	uint32_t value_upper = 0;
	//	int32_t value_multiplication = 1;
	//	int32_t value_division = 1;
	//	int32_t value_add = 0;
	//};

	//struct DataRef
	//{
	//	uint32_t root_variable;
	//	uint32_t root_type;
	//	uint32_t member_type;
	//	std::vector<AccessChainElement> access_chain;
	//};

	//struct Assignment
	//{
	//	DataRef from;
	//	DataRef to;
	//};

	//struct Operatation
	//{
	//	std::vector<Assignment> assignments;
	//};

	//struct ComplexOperatation
	//{
	//	UInt32TOptional current_block_label;

	//	std::unordered_map<uint32_t, std::vector<DataRef>> predessor_to_destinations;
	//	std::unordered_map<uint32_t, std::vector<DataRef>> predessor_to_data;

	//	std::vector<std::pair<AccessChainElement, AccessChainElement>> run_time_loops;
	//};

	// Returns the equiverlant statements for the very complex case when:
	// - is_needing_dynamic_loop is true (it'll give the for loop bounds)
	// - is_needing_branch_write_destinations is true (it'll give the if conditions)
	// - (or both are true - that's a nasty nested case. Fun!)
	//ComplexOperatation equiverlant_multi_block_code(const Instruction &Instruction);

	//// Call this if we only need a single block (see above), and is_needing_multiple_assignments
	//// is true. This will give the series of statements needed for this code.
	//Operatation equiverlant_code_block(const Instruction &Instruction);

	//// Call this in the remaining cases, which are all single statements.
	//Assignment equiverlant_statement(const Instruction &Instruction);

protected:
	Compiler &compiler;
	ParsedIR &ir;

	uint32_t type_of_temporary_id(uint32_t) const;

	// Populate the entire heap memory tree's children with details. This includes
	// identifying pointers in the heap.
	void recurse_populate_memory_tree(TreeNode *Node);

	// See if we can get a perfect, uncasted, aligned, read of memory.
	TreeNode *try_aligned_uncasted_memory_access(const SPIRType &DesiredType, uint32_t Address, uint32_t Size);

	// See if we can get a read of memory but need casts.
	std::vector<TreeNode *> try_aligned_access_memory(const SPIRType &DesiredType, uint32_t Address, uint32_t Size);

	// Try to find the tree block best containing memory. To implement a read / write to this
	// memory we may need to write casts, shifts, float assembly / disassembly / and other
	// horrible things.
	TreeNode *find_memory_location(uint32_t Address, uint32_t Size);

  // Find the "root" allocation, ie the "OpVariable"
	TreeNode *find_memory_root_allocation(uint32_t Address);

  // Find the "leaf" allocation, ie the lowest member.
	TreeNode *find_memory_leaf_allocation(uint32_t Address);

	// Called multiple times, returns true if the potential pointer set grew from
	// detecting pointers being casted to ints, those ints modified, and then
	// casted back to pointers. Also detects when a non-pointer (and non constexpr
	// 0) value is stored in an integer previously identified as also holding a
	// pointer.
	bool find_and_track_pointers(const SPIRBlock &Block);

	// Called once per block once all pointers (and int's that are really pointers) are known,
	// different behaviour caused by Phi nodes changing values is processed here too. Returns
	// whether anything was learnt.
	bool process(const SPIRBlock &Block, uint32_t Precessor);

	// Decodes an access chain
	std::pair<uint32_t, uint32_t> access_chain_to_byte_offset_and_size(const SPIRType &Type, uint32_t ThisStep,
	                                                                   const uint32_t *NextStep,
	                                                                   const uint32_t *EndOfSteps);

	//DataRef address_last_write_ref(TreeNode *Address, uint32_t CurrentBlock, uint32_t Predesessor) const;
	//DataRef temporary_to_ref(uint32_t Id);

	// Simulates operations - returns true if anything new was learnt.
	bool simulate_memcopy(uint32_t SourceAddress, uint32_t DestinationAddress, uint32_t SizeInBytes,
	                      uint32_t ActiveBlock, uint32_t PrecedingBlock);
	bool simulate_store(uint32_t SourceId, uint32_t DestinationAddress, uint32_t SizeInBytes, uint32_t ActiveBlock,
	                    uint32_t PrecedingBlock);

	std::unordered_map<uint32_t, std::shared_ptr<Data>> id_data;

  std::unordered_set<uint32_t> variables_to_cull;
};

} // namespace spirv_cross

#endif
