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

#include "spirv_memory_analyser.hpp"

#include "spirv_arguments.hpp"

using namespace spirv_cross;

uint32_t spirv_cross::MemoryAnalyser::size_of(const SPIRType &Type) const
{
	if (Type.pointer)
	{
		// Pointers have a fixed size, regardless of what they're pointing to.
		return ptr_width;
	}

	if (!Type.array.empty())
	{
		// For arrays, we can use ArrayStride to get an easy check.
		bool array_size_literal = Type.array_size_literal.back();
		uint32_t array_size =
		    array_size_literal ? Type.array.back() : compiler.get<SPIRConstant>(Type.array.back()).scalar();

		uint32_t stride = 0;
		// The stride is hidden, as "Type.self" is unset in the processing of
		// OpTypeArray. There's comments in spirv_parser saying "Do NOT set
		// arraybase.self!". So rather than break whatever that's about, we search
		// through the IR for the duplicated 'self' entries, and take the one with
		// the array stride decoration.
		auto aparentSelf = Type.self;
		for (auto id = 0u; id < ir.ids.size(); id++)
		{
			if (ir.ids[id].get_id() == aparentSelf)
			{
				// id aliases aparentSelf.
				if (ir.meta[id].decoration.array_stride > 0)
				{
					stride = ir.meta[id].decoration.array_stride;
					return stride * array_size;
				}
			}
		}
	}

	if (Type.basetype == SPIRType::Struct)
	{
		return compiler.get_declared_struct_size(Type);
	}

	auto vecsize = Type.vecsize;
	auto columns = Type.columns;

	if (columns == 1)
	{
		size_t component_size = Type.width / 8;
		return vecsize * component_size;
	}
	else
	{
		uint32_t matrix_stride = compiler.get_decoration(Type.self, spv::DecorationMatrixStride);

		if (compiler.has_decoration(Type.self, spv::DecorationRowMajor))
			return matrix_stride * vecsize;
		else if (compiler.has_decoration(Type.self, spv::DecorationColMajor))
			return matrix_stride * columns;
		else
			SPIRV_CROSS_THROW("Either row-major or column-major must be declared for matrices.");
	}
}

GlobalPointerComplexity MemoryAnalyser::process()
{
	memory_tree.clear();
	stack_pointer_details.clear();
	heap_pointer_details.clear();

	// Having 'nullptr' be a valid address just feels wrong, so start a few bytes in.
	auto address = 64u;

	// Create a top level memory tree, placing all known OpVariable data.
	for (auto id : ir.ids)
	{
		auto var = compiler.maybe_get<SPIRVariable>(id.get_id());
		if (var)
		{
			auto ptrType = compiler.get<SPIRType>(var->basetype);

			auto type = compiler.get<SPIRType>(ptrType.parent_type);

			// Place the allocation in memory
			TreeNode node;
			node.address = address;
			node.size_in_bytes = size_of(type);
			node.type = type;
			node.has_parent = false;
			node.owning_variable_id = id.get_id();

			// Register the opVariable as pointing to that memory.
			PointerDetails pd;
			pd.type = type;
			PointerDetails::Address addressInfo;
			addressInfo.exact_address = address;
			pd.addresses = { addressInfo };

			stack_pointer_details[node.owning_variable_id] = std::move(pd);

			// Increment the address of our next allocation, but add between 1 and 64
			// bytes of padding, so we have a nice alignment, and fences.
			address += node.size_in_bytes;
			address = (address + 64) & ~63;

			memory_tree.push_back(std::move(node));
		}
	}

	// Now fill in the details all the way down the tree - so array of struct of
	// some vectors of floats ends up being floats sequentially laid out in memory
	// (assuming all the member offsets and array strides are correct).
	//
	// This also identifies all explicit pointer locations in memory, so we can
	// track their child expressions.
	for (auto &node : memory_tree)
	{
		recurse_memory_tree(&node);
	}

	// Now trace all pointers. We have to repeat this process in case we have code
	// like (Excuse my mix of C++, glsl, and SpirV here):
	//
	//  size_t foo(size_t k) { return k + sizeof(float);}
	//
	//  void main()
	//  {
	//  Entry:
	//    float b[2] = {1.0, 0.5};
	//    size_t c;
	//    size_t d;
	//    float* e;
	//    Goto Work;
	//  Out:
	//    gl_FragColour = vec4(*e, b[0], b[1], b[0]);
	//    return;
	//  Work:
	//    c = reinterpret_cast<size_t>(b);
	//    d = foo(c);
	//    e = reinterpret_cast<float*>(c);
	//    goto Out;
	//  }
	//
	// We don't know that 'c' and d are really a pointer until we process the Work block, and see that
	// it gets the result of an OpPtrToU or OpBitCast instruction, and we wont trace the output of foo
	// until we see that k could be a pointer, which we wont know until later.
	{
		bool foundAPointer = true;
		while (foundAPointer)
		{
			foundAPointer = false;

			for (auto id : ir.ids)
			{
				auto function = compiler.maybe_get<SPIRFunction>(id.get_id());
				if (function)
				{
					std::unordered_set<uint32_t> seenBlocks;
					compiler.function_cfgs[id.get_id()]->walk_from(
					    seenBlocks, function->entry_block, [&](uint32_t block) {
						    foundAPointer |= find_and_track_pointers(compiler.get<SPIRBlock>(block));
					    });
				}
			}
		}
	}

	// Now step through the code, tracking all potential reads and writes, etc.
	for (auto id : ir.ids)
	{
		auto function = compiler.maybe_get<SPIRFunction>(id.get_id());
		if (function)
		{
			std::unordered_set<uint32_t> seenBlocks;
			compiler.function_cfgs[id.get_id()]->walk_from(
			    seenBlocks, function->entry_block, [&](uint32_t block) { process(compiler.get<SPIRBlock>(block)); });
		}
	}

	// Iterate over our state, and try to simplify out as much branching cases in
	// the memory map as possible.

	// Now look at temporary non-scalar non-heap object creation (eg
	// OpComposite<vec>, etc.) by tracing back through shuffles, inserts,
	// extracts, etc back to our now simplified memory access, to see if we can
	// forward what we've learnt from our memory analysis further out into the
	// code - beyond the OpLoad. Hopefully turning the string of loads,
	// BitCasts, Copies, shuffles, and inserts (typical result from an SSE / AVX /
	// etc optimisation engine) into a single clear glsl instruction. The goal is
	// to be able to initialise a vector or other composite type from a collection
	// of lvalues referencing directly into the structured blob the data came
	// from, without temporaries.

	// Now try to work out whether our analysis steps above removed the need for
	// some instructions entirely, and these could be skipped entirely from the
	// glsl export pass.

	//

	return GlobalPointerComplexity::AllTrivial;
}

uint32_t spirv_cross::MemoryAnalyser::type_of_temporary_id(uint32_t Id) const
{
	// Try some easy cases:

	auto expression = compiler.maybe_get<SPIRExpression>(Id);

	if (expression)
	{
		return expression->expression_type;
	}

	auto variable = compiler.maybe_get<SPIRVariable>(Id);
	if (variable)
	{
		return variable->basetype;
	}

	// Ok - try and find the instruction which creates it, as that'll declare it's type.
	for (uint32_t maybeBlock = 0; maybeBlock < ir.ids.size(); maybeBlock++)
	{
		auto block = compiler.maybe_get<SPIRBlock>(maybeBlock);

		if (!block)
			continue;

		for (auto &instruction : block->ops)
		{
			if (has_result(spv::Op(instruction.op)))
			{
				auto args = compiler.stream(instruction);
				return args[0];
			}
		}
	}

	assert(false);
	return 0;
}

void spirv_cross::MemoryAnalyser::recurse_memory_tree(TreeNode *Node)
{
	if (Node->type.pointer)
	{
		// This is a pointer stored in heap memory. Add it to our pointer tracking.

		auto &pd = heap_pointer_details[Node->address];
		pd.type = compiler.get<SPIRType>(Node->type.parent_type);

		return;
	}

	if (Node->type.basetype == SPIRType::Struct)
	{
		Node->has_child = true;

		// We have to iterate through the struct members and place them in memory.
		// Luckily there is the Offset decoration to make this super easy.

		for (auto childIndex = 0u; childIndex < Node->type.member_types.size(); childIndex++)
		{
			TreeNode node;
			node.type = compiler.get_type(Node->type.member_types[childIndex]);
			auto offset = compiler.get_member_decoration(Node->type.self, childIndex, spv::DecorationOffset);
			auto size = compiler.get_declared_struct_member_size(Node->type, childIndex);

			node.address = Node->address + offset;
			node.size_in_bytes = size;
			node.has_parent = true;
			node.local_access_chain = Node->local_access_chain;
			node.local_access_chain.push_back(childIndex);

			Node->children.push_back(std::move(node));
		}
	}
	else if (Node->type.array.size())
	{
		Node->has_child = true;

		TreeNode node;
		node.type = compiler.get_type(Node->type.parent_type);

		auto childSize = size_of(node.type);
		node.size_in_bytes = childSize;
		node.has_parent = true;

		auto arraySize = size_of(Node->type);
		auto stride = arraySize / Node->type.array.back();

		node.local_access_chain = Node->local_access_chain;
		node.local_access_chain.push_back(0);

		for (auto childIndex = 0u; childIndex < Node->type.array.back(); childIndex++)
		{
			node.address = Node->address + childIndex * stride;
			node.local_access_chain.back() = childIndex;

			Node->children.push_back(node);
		}
	}
	else if (compiler.has_decoration(Node->type.self, spv::DecorationMatrixStride))
	{
		uint32_t matrix_stride = compiler.get_decoration(Node->type.self, spv::DecorationMatrixStride);
		Node->has_child = true;
		uint32_t childCount;
		//uint32_t childStride;
		if (compiler.has_decoration(Node->type.self, spv::DecorationRowMajor))
		{
			childCount = Node->type.vecsize;
		}
		else
		{
			childCount = Node->type.columns;
		}

		TreeNode node;
		node.local_access_chain = Node->local_access_chain;
		node.local_access_chain.push_back(0);
		node.type = compiler.get_type(Node->type.parent_type);
		node.has_parent = true;
		node.size_in_bytes = size_of(node.type);

		for (auto childIndex = 0u; childIndex < childCount; childIndex++)
		{
			node.address = Node->address + childIndex * node.size_in_bytes;
			node.local_access_chain.back() = childIndex;

			Node->children.push_back(node);
		}
	}
	else if (Node->type.vecsize > 1)
	{
		Node->has_child = true;

		TreeNode node;
		node.local_access_chain = Node->local_access_chain;
		node.local_access_chain.push_back(0);
		node.type = compiler.get_type(Node->type.parent_type);
		node.has_parent = true;
		node.size_in_bytes = size_of(node.type);

		for (auto childIndex = 0u; childIndex < Node->type.vecsize; childIndex++)
		{
			node.address = Node->address + childIndex * node.size_in_bytes;
			node.local_access_chain.back() = childIndex;

			Node->children.push_back(node);
		}
	}
	else
	{
		// Scalar type.
		return;
	}

	if (Node->has_child)
	{
		for (auto &child : Node->children)
		{
			recurse_memory_tree(&child);
		}
	}
}

bool spirv_cross::MemoryAnalyser::find_and_track_pointers(const SPIRBlock &Block)
{
	bool learntSomething = false;
	std::vector<uint32_t> pointersIn;
	std::vector<uint32_t> nonPointerIdsIn;
	std::vector<uint32_t> litteralsIn;

	for (auto &instruction : Block.ops)
	{
		pointersIn.clear();
		nonPointerIdsIn.clear();
		litteralsIn.clear();
		auto *ops = compiler.stream(instruction);
		auto op = static_cast<spv::Op>(instruction.op);

		for (auto arg = 0; arg < instruction.count; arg++)
		{
			if (is_input_id(op, arg))
			{
				auto it = this->stack_pointer_details.find(ops[arg]);
				if (it != this->stack_pointer_details.end())
				{
					// We passed a pointer into this instruction.
					pointersIn.push_back(ops[arg]);
				}
				else
				{
					nonPointerIdsIn.push_back(ops[arg]);
				}
			}
			else
			{
				litteralsIn.push_back(ops[arg]);
			}
		}

		// We can ignore if it takes no pointers or pointer-like-ints.
		if (pointersIn.empty())
			continue;

		auto morphPointerNoOp = [&]() {
			if (has_result(op) && pointersIn.size() == 1)
			{
				auto returnTypeNo = ops[0];
				auto returnidNo = ops[1];
				auto returnType = compiler.get<SPIRType>(returnTypeNo);

				auto &ptrInfo = stack_pointer_details[returnidNo];
				auto &inputInfo = stack_pointer_details[pointersIn.front()];

				if (ptrInfo.addresses.empty())
				{
					// We've learnt something here!
					ptrInfo = stack_pointer_details[pointersIn.front()];
					learntSomething = true;
				}
			}
			else
			{
				SPIRV_CROSS_THROW("Instruction took a pointer and didn't output anything, or took two pointers.");
			}
		};

		auto morphPointerAddConstant = [&](uint32_t offset) {
			if (has_result(op) && pointersIn.size() == 1)
			{
				auto returnTypeNo = ops[0];
				auto returnidNo = ops[1];
				auto returnType = compiler.get<SPIRType>(returnTypeNo);

				auto &ptrInfo = stack_pointer_details[returnidNo];
				auto &inputInfo = stack_pointer_details[pointersIn.front()];

				if (ptrInfo.addresses.empty())
				{
					// We've learnt something here!
					ptrInfo.addresses.push_back({});
					ptrInfo.addresses.back().base_variable_id = pointersIn.front();
					ptrInfo.addresses.back().litteral_offset = int32_t(offset);

					learntSomething = true;
				}
			}
			else
			{
				SPIRV_CROSS_THROW("Instruction took a pointer and didn't output anything, or took two pointers.");
			}
		};

		switch (op)
		{
		case spv::OpStore:
		{
			// OpStore with a single pointer arg can be ignored most of the time (for
			// the purposes of pointer tracking). If it's writing a pointer to memory,
			// or writing to memory that contains a pointer, that's a more complicated
			// case.
			//:TODO: Detect if storing a pointer in the heap.
			break;
		}
		case spv::OpLoad:
		{
			// OpLoad on a pointer doesn't (often) return a pointer.
			//:TODO: detect if loading a pointer stored in the heap.
			break;
		}
		case spv::OpBitcast:
		{
			// OpBitcast just changes the type of the pointer, or converts it to or
			// from an integer. The validation rules insist that they must have the
			// same number of bits, so it's just a no-op.
			morphPointerNoOp();
			break;
		}
		case spv::OpConvertPtrToU:
		{
			auto returnTypeNo = ops[0];
			auto typeInfo = compiler.get<SPIRType>(returnTypeNo);

			if (typeInfo.width == ptr_width * 8)
			{
				// Direct pointer cast to int - no op, and bless the integer as a pointer.
				morphPointerNoOp();
			}
			else
			{
				// This is technically valid SPIR-V, but it just seems so dubious, and unsafe.
				//
				// (I haven't seen this in the wild.)
				SPIRV_CROSS_THROW("Truncation of memory address to a smaller integer type.");
			}
			break;
		}

		case spv::OpConvertUToPtr:
		case spv::OpGenericCastToPtr:
		case spv::OpGenericCastToPtrExplicit:
		case spv::OpPtrCastToGeneric:
		{
			// These pointer casts are always 'no-ops' - just forward the pointer.
			morphPointerNoOp();
			break;
		}

		case spv::OpUConvert:
		case spv::OpSConvert:
		{
			auto returnTypeNo = ops[0];
			auto typeInfo = compiler.get<SPIRType>(returnTypeNo);
			if (typeInfo.width >= ptr_width * 8)
			{
				// Casting up a pointer to a larger type... weird, but legal and should be
				// workable.
				morphPointerNoOp();
			}
			else
			{
				// This is technically valid SPIR-V, but it just seems so dubious, and unsafe.
				SPIRV_CROSS_THROW("Truncation of memory address to a smaller integer type.");
			}

			break;
		}
		case spv::OpFConvert:
		case spv::OpConvertUToF:
		case spv::OpConvertSToF:
		{
			SPIRV_CROSS_THROW("Converting a memory address to a floating point value is absurd.");
		}

		case spv::OpIAdd:
		{
			// :TODO: find the non-pointer, and track it back to a constant if possible, else try to
			// derive an expression for it.
			break;
		}

		case spv::OpISub:
		{
			// Two cases - either:
			// - diff between two pointers, which is later used for something, eg
			//   convert an anonymous access chain into a pointer diff, and then apply
			//   it to any pointer.
			// - Or moving backwards through an array. ew.
			//
			// :TODO: find the non-pointer, and track it back to a constant if
			// possible, else try to derive an expression for it.
			break;
		}

		case spv::OpUMod:
		case spv::OpUDiv:
		case spv::OpBitwiseAnd:
		case spv::OpBitwiseOr:
		case spv::OpBitwiseXor:
		case spv::OpShiftLeftLogical:
		case spv::OpShiftRightArithmetic:
		case spv::OpShiftRightLogical:
		{
			// This may be needed for a compilation of something like a boost::intrusive::set or some other
			// non allocating rbtree with an optimisation which stores the red / black flag within the lsb of the pointer.
			SPIRV_CROSS_THROW("Bit manipulation on pointers not yet implemented.");
		}

		case spv::OpInBoundsAccessChain:
		case spv::OpAccessChain:
		{
			// We need to step into memory following the indices to calculate an offset.

			// This is basically the C++ arrow operator. These are the same as OpPtrAccessChain and
			// OpInBoundsPtrAccessChain, but with an extra '0' at the start. :-)
			// (See https://llvm.org/docs/GetElementPtr.html)

			auto& inputTypePtr = compiler.get<SPIRType>(this->type_of_temporary_id(ops[2]));

      auto& inputType = compiler.get<SPIRType>(inputTypePtr.parent_type);

			auto offsetAndSize = access_chain_to_byte_offset_and_size(inputType, 0, ops + 3, ops + instruction.count - 1);

			auto outputTypePtr = ops[0];
			auto &outputType = compiler.get<SPIRType>(compiler.get<SPIRType>(outputTypePtr).parent_type);

			auto outputSize = size_of(outputType);

			if (outputSize != offsetAndSize.second)
			{
				SPIRV_CROSS_THROW("Access chain size mismatch?");
			}

			morphPointerAddConstant(offsetAndSize.first);

			break;
		}
		case spv::OpPtrAccessChain:
		case spv::OpInBoundsPtrAccessChain:
		{
			// We need to step into memory following the indices to calculate an offset.
			// This is basically a C++ square bracket following a pointer.
			// (See https://llvm.org/docs/GetElementPtr.html)

			auto inputType = compiler.get<SPIRType>(this->type_of_temporary_id(ops[2]));

			auto offsetAndSize = access_chain_to_byte_offset_and_size(inputType, 0, ops + 3, ops + instruction.count - 1);

			auto outputTypePtr = ops[0];
			auto &outputType = compiler.get<SPIRType>(compiler.get<SPIRType>(outputTypePtr).parent_type);

			auto outputSize = size_of(outputType);

			if (outputSize != offsetAndSize.second)
			{
				SPIRV_CROSS_THROW("Access chain size mismatch?");
			}

			morphPointerAddConstant(offsetAndSize.first);

			break;
		}
		default:
			SPIRV_CROSS_THROW("Pointer passed into a not yet implemented instruction");
		}
	}

	return learntSomething;
}

void spirv_cross::MemoryAnalyser::process(const SPIRBlock &Function)
{
}

std::pair<uint32_t, uint32_t> spirv_cross::MemoryAnalyser::access_chain_to_byte_offset_and_size(
    const SPIRType &Type, uint32_t ThisStep, const uint32_t *NextStep, const uint32_t *EndOfSteps)
{
  if (NextStep == EndOfSteps)
  {
    auto s = size_of(Type);
		return { uint32_t(s * ThisStep), s };
  }

	if (Type.pointer)
	{
		auto &pointedType = compiler.get<SPIRType>(Type.parent_type);
		auto sizeOfPointed = size_of(pointedType);

		auto out = access_chain_to_byte_offset_and_size(pointedType, compiler.get_constant(*NextStep).scalar_i32(),
		                                                NextStep + 1, EndOfSteps);
		out.first += ThisStep * sizeOfPointed;
		return out;
	}
	else if (Type.array.size())
	{
		auto &arrayElement = compiler.get<SPIRType>(Type.parent_type);
		auto sizeOfElement = size_of(arrayElement);

		auto arraySize = size_of(Type);
		auto stride = arraySize / Type.array.back();

		auto out = access_chain_to_byte_offset_and_size(arrayElement, compiler.get_constant(*NextStep).scalar_i32(),
		                                                NextStep + 1, EndOfSteps);
		out.first += ThisStep * stride;
		return out;
	}
	else if (Type.basetype == Type.Struct)
	{
		auto &childType = compiler.get<SPIRType>(Type.member_types[ThisStep]);
		auto childOffset = compiler.type_struct_member_offset(Type, ThisStep);

		auto out = access_chain_to_byte_offset_and_size(childType, compiler.get_constant(*NextStep).scalar_i32(),
		                                                NextStep + 1, EndOfSteps);
		out.first += childOffset;

		return out;
	}
	else if (Type.columns > 1)
	{
		// We're a matrix

		//:TODO: Figure out row major / column major stuff.
		return {};
	}
	else if (Type.vecsize > 1)
	{
		// We're a vector.

		auto &childElement = compiler.get<SPIRType>(Type.parent_type);
		auto sizeOfElement = size_of(childElement);

		auto vectorSize = size_of(Type);
		auto stride = vectorSize / Type.array.back();

		auto out = access_chain_to_byte_offset_and_size(childElement, *NextStep, NextStep + 1, EndOfSteps);
		out.first += ThisStep * stride;

		return out;
	}
	else
	{
		SPIRV_CROSS_THROW("Stepping into a scalar");
	}
}
