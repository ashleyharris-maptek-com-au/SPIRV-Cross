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

#include <future>
#include <numeric>

using namespace spirv_cross;

std::vector<uint32_t> spirv_cross::MemoryAnalyser::all_potential_stack_pointer_destinations(
    uint32_t Id, UInt32TOptional LabelOfQueryingBlock, UInt32TOptional LabelOfPrecessorBlock) const
{
	//:TODO: if code called set_hint_ptr_bounds, set_hint_ptr_address, etc, use that.

	auto it = stack_pointer_details.find(Id);
	if (it == stack_pointer_details.end())
	{
		SPIRV_CROSS_THROW("Not a pointer?");
	}

	std::vector<uint32_t> outputs;

	auto processAddress = [&](const PointerDetails::Address &A) {
		//:TODO: if set_in_block_label && potentialCfg.contains(set_in_block_label) && potentialCfg.conditional_predecessor_label, etc. etc.

		if (A.exact_address)
		{
			outputs.push_back(A.exact_address);
		}
		else if (A.base_variable_id)
		{
			auto addresses = all_potential_stack_pointer_destinations(A.base_variable_id, LabelOfQueryingBlock,
			                                                          LabelOfPrecessorBlock);

			std::vector<uint32_t> offsets;
			if (A.offset_id)
			{
				offsets = all_potential_integer_values(A.offset_id, LabelOfQueryingBlock, LabelOfPrecessorBlock);
			}
			else
			{
				if (A.litteral_offset < 0)
					SPIRV_CROSS_THROW("NYI");
				offsets = { uint32_t(A.litteral_offset) };
			}

			for (auto ptr : addresses)
			{
				for (auto offset : offsets)
				{
					outputs.push_back(ptr + offset);
				}
			}
		}
	};

	if (it->second.addresses.size() == 1)
	{
		processAddress(it->second.addresses.front());
	}
	else
	{
		SPIRV_CROSS_THROW("NYI - we need to search back through the cfg to answer this");
		// processAddress(it->second.addresses....);
	}

	return outputs;
}

std::vector<uint32_t> spirv_cross::MemoryAnalyser::all_potential_integer_values(
    uint32_t Id, UInt32TOptional LabelOfQueryingBlock, UInt32TOptional LabelOfPrecessorBlock) const
{
	//:TODO: if user called set_hint_int_bounds, set_hint_int_value, etc, use that.

	auto *maybeConstant = compiler.maybe_get<SPIRConstant>(Id);
	if (maybeConstant)
	{
		//:TODO: This wont handle crazy cases, like:
		//
		//   volatile uint64_t w = 1'234'567'890'123'546ull;
		//   uint32_t u = *((&v) + (w >> 40ull))
		//
		// This is meant to decode inputs into pointer arithmetic, so they're ideally
		// either trivial or impossible to statically analyses.
		return { uint32_t(maybeConstant->scalar_u64()) };
	}

	auto *maybeExpression = compiler.maybe_get<SPIRExpression>(Id);
	if (maybeExpression)
	{
		SPIRV_CROSS_THROW("TODO");
	}

	SPIRV_CROSS_THROW("NYI");
}

std::vector<uint32_t> spirv_cross::MemoryAnalyser::all_potential_sensible_mem_copy_sizes(
    uint32_t SizeId, uint32_t SourcePtrAddress, uint32_t DestPtrAddress, UInt32TOptional LabelOfQueryingBlock,
    UInt32TOptional LabelOfPrecessorBlock)
{
	SPIRV_CROSS_THROW("NYI");

	std::vector<uint32_t> potentials;

	potentials = all_potential_integer_values(SizeId, LabelOfQueryingBlock, LabelOfPrecessorBlock);

	auto block1 = find_memory_root_allocation(SourcePtrAddress);
	auto block2 = find_memory_root_allocation(DestPtrAddress);

	auto maxSize = std::min(block1->size_in_bytes, block2->size_in_bytes);

	std::vector<uint32_t> output;

	for (auto offset = 0u; offset < maxSize; offset++)
	{
		//auto b1 = find_memory_leaf_allocation()
	}

	return {};
}

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

std::vector<uint32_t> spirv_cross::MemoryAnalyser::child_types(const SPIRType &Type) const
{
	if (Type.pointer)
	{
		// Pointers are scalars.
		return {};
	}

	if (!Type.array.empty())
	{
		bool array_size_literal = Type.array_size_literal.back();
		uint32_t array_size =
		    array_size_literal ? Type.array.back() : compiler.get<SPIRConstant>(Type.array.back()).scalar();

		return std::vector<uint32_t>(array_size, Type.parent_type);
	}

	if (Type.basetype == SPIRType::Struct)
	{
		return Type.member_types;
	}

	auto vecsize = Type.vecsize;
	auto columns = Type.columns;

	if (vecsize == 1 && columns == 1)
		return {};

	if (columns == 1)
	{
		return std::vector<uint32_t>(vecsize, Type.parent_type);
	}
	else
	{
		uint32_t matrix_stride = compiler.get_decoration(Type.self, spv::DecorationMatrixStride);

		if (compiler.has_decoration(Type.self, spv::DecorationRowMajor))
			return std::vector<uint32_t>(vecsize, Type.parent_type);
		else if (compiler.has_decoration(Type.self, spv::DecorationColMajor))
			return std::vector<uint32_t>(columns, Type.parent_type);
		else
			SPIRV_CROSS_THROW("Either row-major or column-major must be declared for matrices.");
	}

	return {};
}

bool spirv_cross::MemoryAnalyser::type_holds_pointer(const SPIRType &Type) const
{
	// NYI
	return false;
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
			node.type = ptrType.parent_type;
			node.has_parent = false;
			node.owning_variable_id = id.get_id();
			node.storage_class = var->storage;

			if (var->storage == spv::StorageClassUniformConstant || var->storage == spv::StorageClassInput ||
			    var->storage == spv::StorageClassUniform || var->storage == spv::StorageClassWorkgroup ||
			    var->storage == spv::StorageClassCrossWorkgroup || var->storage == spv::StorageClassGeneric ||
			    var->storage == spv::StorageClassPushConstant || var->storage == spv::StorageClassAtomicCounter ||
			    var->storage == spv::StorageClassImage || var->storage == spv::StorageClassStorageBuffer)
			{
				// These values have data set externally, which we must represent in our memory tree.
				struct DataExternal : Data
				{
					uint32_t data;
					Resolution resolved(UInt32TOptional CurrentBlock = {},
					                    UInt32TOptional PrecedingBlock = {}) const override
					{
						return ExistingId;
					}

					uint32_t id(UInt32TOptional CurrentBlock = {}, UInt32TOptional PrecedingBlock = {}) const override
					{
						return data;
					}

					bool needs_backing_variable() const override
					{
						return false;
					}
				};

				auto value = std::make_shared<DataExternal>();
				value->data = node.owning_variable_id;
				node.value = value;
			}

			if (var->initializer)
			{
				// We have an initialiser - we should note it.
				if (node.value)
				{
					// We already have defined a value here - ie it's external memory. This means the initialiser is a default value,
					// eg a uniform with a fallback. We can't know the value of the memory.
				}
				else
				{
					auto constant = std::make_shared<DataConstant>();
					auto maybeConstant = compiler.maybe_get<SPIRConstant>(var->initializer);
					auto maybeConstantOp = compiler.maybe_get<SPIRConstantOp>(var->initializer);

					if (maybeConstant)
					{
						auto constantType = compiler.get_type(maybeConstant->constant_type);
						constant->isScalar = (constantType.parent_type == 0);
						constant->valueConstant = maybeConstant;
						constant->type = maybeConstant->constant_type;
					}
					else
					{
						constant->valueConstExpr = maybeConstantOp;
						constant->type = maybeConstantOp->basetype;
					}
				}
			}

			// Register the value returned by opVariable as pointing to that memory.
			PointerDetails pd;
			pd.type = ptrType.parent_type;
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
		recurse_populate_memory_tree(&node);
	}

	std::vector<SPIRFunction *> functions;

	for (auto id : ir.ids)
	{
		auto function = compiler.maybe_get<SPIRFunction>(id.get_id());
		if (function)
			functions.push_back(function);
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

			for (auto function : functions)
			{
				std::unordered_set<uint32_t> seenBlocks;
				compiler.function_cfgs[function->self]->walk_from(
				    seenBlocks, function->entry_block,
				    [&](uint32_t block) { foundAPointer |= find_and_track_pointers(compiler.get<SPIRBlock>(block)); });
			}
		}
	}

	// Now step through the code, tracking all potential reads and writes, etc.
	uint32_t passCounter = 0;
	do
	{
		passCounter++;
		bool learntAnything = false;
		for (auto function : functions)
		{
			std::vector<uint32_t> blockQueue = function->blocks;
			std::sort(blockQueue.begin(), blockQueue.end());

			auto &cfg = compiler.function_cfgs[function->self];

		makeThisAWhileLoop:

			bool hasMadeProgress = false;

			for (auto block : blockQueue)
			{
				bool allEntriesProcessed = true;

				if (cfg->get_preceding_edges(block).empty())
				{
					// We're the function entry.
					process(compiler.get<SPIRBlock>(block), uint32_t(-1));
				}
				else
				{
					// We're a conditional / looping block. Make sure all blocks we can come from
					// have already been processed.
					for (auto &entry : cfg->get_preceding_edges(block))
					{
						if (std::binary_search(blockQueue.begin(), blockQueue.end(), entry))
						{
							allEntriesProcessed = false;
							break;
						}
					}
				}

				if (!allEntriesProcessed)
					continue;

				blockQueue.erase(std::remove(blockQueue.begin(), blockQueue.end(), block));
				hasMadeProgress = true;

				// Now process each place we can go to, mentioning we came from here.
				for (auto &exit : cfg->get_succeeding_edges(block))
				{
					process(compiler.get<SPIRBlock>(exit), block);
				}

				break;
			}

			if (!hasMadeProgress)
			{
				// We're stuck in a loop, potentially infinite depth expression tree.
				//
				// Firstly - try to find blocks with no memory access, if such a block
				// exists, then we can mark it as processed, and that might unblock us.
				//
				// Otherwise, we may need to require hints to resolve this, or to give up
				// and produce suboptimal code.
				//
				SPIRV_CROSS_THROW("NYI");
			}

			if (blockQueue.size())
				goto makeThisAWhileLoop;
		}
		if (!learntAnything)
			break;
	} while (true);

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

	std::unordered_set<uint32_t> rootAddressesNeeded;

	for (auto function : functions)
	{
		std::unordered_set<uint32_t> seenBlocks;
		compiler.function_cfgs[function->self]->walk_from(seenBlocks, function->entry_block, [&](uint32_t block) {
			for (auto &op : compiler.get<SPIRBlock>(block).ops)
			{
				if (op.op == spv::OpLoad)
				{
					auto args = compiler.stream(op);
					auto returnValue = args[1];

					auto ptr = args[2];

					auto &data = id_data[returnValue];

					if (data->needs_backing_variable())
					{
						auto addresses = all_potential_stack_pointer_destinations(ptr);

						for (auto &address : addresses)
						{
							auto root = this->find_memory_root_allocation(address);
							rootAddressesNeeded.insert(root->address);
						}
					}
				}
			}
		});
	}

	for (auto &tree : memory_tree)
	{
		if ((tree.storage_class == spv::StorageClassFunction || tree.storage_class == spv::StorageClassPrivate) &&
		    rootAddressesNeeded.find(tree.address) == rootAddressesNeeded.end())
		{
			variables_to_cull.insert(tree.owning_variable_id);
		}
	}

	// So now we can freely erase any function / private variable access that
	// isn't within a rootVariablesNeeded ancestor. as well as any operation that
	// depends on it, making sure to replace the OpLoad(s) with the data we
	// discovered for it.
	std::unordered_set<uint32_t> redundantPointers;
	for (auto &pointer : stack_pointer_details)
	{
		auto addresses = all_potential_stack_pointer_destinations(pointer.first);

		bool allRedundant = true;
		for (auto &address : addresses)
		{
			auto *rootAllocation = find_memory_root_allocation(address);

			if (rootAllocation->storage_class != spv::StorageClassFunction &&
			    rootAllocation->storage_class != spv::StorageClassPrivate)
			{
				// Someone else can set this memory at any time.
				allRedundant = false;
				break;
			}

			if (rootAddressesNeeded.find(rootAllocation->address) != rootAddressesNeeded.end())
			{
				// We need this allocation.
				allRedundant = false;
				break;
			}
		}

		if (allRedundant)
			redundantPointers.insert(pointer.first);
	}

	for (auto function : functions)
	{
		for (auto blockId : function->blocks)
		{
			for (auto &instruction : compiler.get<SPIRBlock>(blockId).ops)
			{
				bool remove = false;
				auto args = compiler.stream(instruction);
				if (has_result(spv::Op(instruction.op)))
				{
					auto returnValue = args[1];
					if (redundantPointers.find(returnValue) != redundantPointers.end())
					{
						// We're returning a redundant pointer - remove.
						remove = true;
					}
				}

				for (auto i = 0u; i < instruction.count; i++)
				{
					if (is_input_id(spv::Op(instruction.op), i))
					{
						if (redundantPointers.find(args[i]) != redundantPointers.end())
						{
							// We're using a redundant pointer - remove.
							remove = true;
						}
					}
				}

				if (remove)
				{
					if (instruction.op == spv::OpLoad)
					{
						// We're loading from memory identified as removable - so we need to replace the
						// load with something a bit more useful.

						instruction.op = OpGenerateViaSpirvCrossMemoryAnalyser;
					}
					else
					{
						instruction.op = spv::OpNop;
						// I wonder if I have to patch the stream?
					}
				}
			}
		}
	}

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

	auto constant = compiler.maybe_get<SPIRConstant>(Id);
	if (constant)
	{
		return constant->constant_type;
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
				if (args[1] == Id)
					return args[0];
			}
		}
	}

	assert(false);
	return 0;
}

void spirv_cross::MemoryAnalyser::recurse_populate_memory_tree(TreeNode *Node)
{
	auto &nodeType = compiler.get_type(Node->type);
	if (nodeType.pointer)
	{
		// This is a pointer stored in heap memory. Add it to our pointer tracking.

		auto &pd = heap_pointer_details[Node->address];
		pd.type = nodeType.parent_type;

		return;
	}

	if (nodeType.basetype == SPIRType::Struct)
	{
		Node->has_child = true;

		// We have to iterate through the struct members and place them in memory.
		// Luckily there is the Offset decoration to make this super easy.

		for (auto childIndex = 0u; childIndex < nodeType.member_types.size(); childIndex++)
		{
			TreeNode node;
			node.type = nodeType.member_types[childIndex];
			auto offset = compiler.get_member_decoration(nodeType.self, childIndex, spv::DecorationOffset);
			auto size = compiler.get_declared_struct_member_size(nodeType, childIndex);

			node.address = Node->address + offset;
			node.size_in_bytes = size;
			node.has_parent = true;
			node.local_access_chain = Node->local_access_chain;
			node.local_access_chain.push_back(childIndex);

			auto genesis = find_memory_root_allocation(node.address);
			if (genesis->value)
			{
				// We're not undef - so set our value as a reference to our ancestors value + access chain.
				auto valueChain = std::make_shared<DataAccessChain>();
				valueChain->access_chain = node.local_access_chain;
				valueChain->parent_value = genesis->value;
				valueChain->relative_pointer = node.address - genesis->address;
				node.value = valueChain;
			}

			Node->children.push_back(std::move(node));
		}
	}
	else if (nodeType.array.size())
	{
		Node->has_child = true;

		TreeNode node;
		node.type = nodeType.parent_type;

		auto childSize = size_of(compiler.get_type(nodeType.parent_type));
		node.size_in_bytes = childSize;
		node.has_parent = true;

		auto arraySize = size_of(nodeType);
		auto stride = arraySize / nodeType.array.back();

		node.local_access_chain = Node->local_access_chain;
		node.local_access_chain.push_back(0);

		for (auto childIndex = 0u; childIndex < nodeType.array.back(); childIndex++)
		{
			node.address = Node->address + childIndex * stride;
			node.local_access_chain.back() = childIndex;

			Node->children.push_back(node);
		}
	}
	else if (compiler.has_decoration(nodeType.self, spv::DecorationMatrixStride))
	{
		uint32_t matrix_stride = compiler.get_decoration(nodeType.self, spv::DecorationMatrixStride);
		Node->has_child = true;
		uint32_t childCount;
		//uint32_t childStride;
		if (compiler.has_decoration(nodeType.self, spv::DecorationRowMajor))
		{
			childCount = nodeType.vecsize;
		}
		else
		{
			childCount = nodeType.columns;
		}

		TreeNode node;
		node.local_access_chain = Node->local_access_chain;
		node.local_access_chain.push_back(0);
		node.type = nodeType.parent_type;
		node.has_parent = true;
		node.size_in_bytes = size_of(compiler.get_type(nodeType.parent_type));

		for (auto childIndex = 0u; childIndex < childCount; childIndex++)
		{
			node.address = Node->address + childIndex * node.size_in_bytes;
			node.local_access_chain.back() = childIndex;

			Node->children.push_back(node);
		}
	}
	else if (nodeType.vecsize > 1)
	{
		Node->has_child = true;

		TreeNode node;
		node.local_access_chain = Node->local_access_chain;
		node.local_access_chain.push_back(0);
		node.type = nodeType.parent_type;
		node.has_parent = true;
		node.size_in_bytes = size_of(compiler.get_type(nodeType.parent_type));

		for (auto childIndex = 0u; childIndex < nodeType.vecsize; childIndex++)
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
			recurse_populate_memory_tree(&child);
		}
	}
}

spirv_cross::MemoryAnalyser::TreeNode *spirv_cross::MemoryAnalyser::try_aligned_uncasted_memory_access(
    const SPIRType &DesiredType, uint32_t Address, uint32_t Size)
{
	auto block = find_memory_location(Address, Size);

	if (block->address == Address && block->size_in_bytes == Size && &compiler.get_type(block->type) == &DesiredType)
	{
		// That was lucky!
		return block;
	}

	return nullptr;
}

std::vector<spirv_cross::MemoryAnalyser::TreeNode *> spirv_cross::MemoryAnalyser::try_aligned_access_memory(
    const SPIRType &DesiredType, uint32_t Address, uint32_t Size)
{
	auto block = find_memory_location(Address, Size);

	auto sizeOf = size_of(DesiredType);

	if (block->size_in_bytes < Size)
	{
		SPIRV_CROSS_THROW("Potential buffer overrun.");
	}

	if (&compiler.get_type(block->type) == &DesiredType)
	{
		// We got a match!
		return { block };
	}

	// We need to try to line up the children
	auto offset = 0;
	auto childTypes = child_types(DesiredType);

	if (childTypes.empty())
	{
		// We're at the base of the type heirchay - hopefully in memory we're at scalar level too.
		if (block->size_in_bytes == sizeOf)
		{
			return { block };
		}
		return {};
	}

	std::vector<spirv_cross::MemoryAnalyser::TreeNode *> output;

	for (auto child : childTypes)
	{
		auto &childType = compiler.get_type(child);
		auto sizeOfChild = size_of(childType);
		auto result = try_aligned_access_memory(childType, Address + offset, sizeOfChild);
		if (result.empty())
			return {};
		output.insert(output.end(), result.begin(), result.end());
		offset += sizeOfChild;
	}

	return output;
}

spirv_cross::MemoryAnalyser::TreeNode *spirv_cross::MemoryAnalyser::find_memory_location(uint32_t Address,
                                                                                         uint32_t Size)
{
	std::vector<TreeNode> *branch = &this->memory_tree;
	TreeNode *output = nullptr;

	while (!branch->empty())
	{
		bool inAChild = false;
		for (auto &child : *branch)
		{
			if (Address >= child.address && Address + Size <= child.address + child.size_in_bytes)
			{
				inAChild = true;
				branch = &child.children;
				output = &child;

				if (branch->empty())
				{
					return &child;
				}
				break;
			}
		}

		if (!inAChild)
		{
			return output;
		}
	}

	return nullptr;
}

spirv_cross::MemoryAnalyser::TreeNode *spirv_cross::MemoryAnalyser::find_memory_root_allocation(uint32_t Address)
{
	for (auto &child : memory_tree)
	{
		if (Address >= child.address && Address <= child.address + child.size_in_bytes)
		{
			return &child;
		}
	}
	return nullptr;
}

spirv_cross::MemoryAnalyser::TreeNode *spirv_cross::MemoryAnalyser::find_memory_leaf_allocation(uint32_t Address)
{
	std::vector<TreeNode> *branch = &this->memory_tree;

	while (!branch->empty())
	{
		for (auto &child : *branch)
		{
			if (Address >= child.address && Address <= child.address + child.size_in_bytes)
			{
				branch = &child.children;
				if (branch->empty())
				{
					return &child;
				}
				break;
			}
		}
	}

	return nullptr;
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
			if (type_holds_pointer(compiler.get<SPIRType>(type_of_temporary_id(nonPointerIdsIn.front()))))
			{
				SPIRV_CROSS_THROW("NYI - pointer in heap tracking");
			}
			break;
		}
		case spv::OpLoad:
		{
			// OpLoad on a pointer doesn't (often) return a pointer.
			if (type_holds_pointer(compiler.get<SPIRType>(ops[0])))
			{
				SPIRV_CROSS_THROW("NYI - pointer in heap tracking");
			}
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
			SPIRV_CROSS_THROW("NYI.");
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
			SPIRV_CROSS_THROW("NYI.");
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

			auto &inputTypePtr = compiler.get<SPIRType>(this->type_of_temporary_id(ops[2]));

			auto &inputType = compiler.get<SPIRType>(inputTypePtr.parent_type);

			auto offsetAndSize =
			    access_chain_to_byte_offset_and_size(inputType, 0, ops + 3, ops + instruction.count - 1);

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

			auto offsetAndSize =
			    access_chain_to_byte_offset_and_size(inputType, 0, ops + 3, ops + instruction.count - 1);

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

bool spirv_cross::MemoryAnalyser::process(const SPIRBlock &Block, uint32_t Precessor)
{
	for (auto &instruction : Block.ops)
	{
		auto ops = compiler.stream(instruction);
		switch (instruction.op)
		{
		case spv::OpCopyMemory:
		{
			auto targetPtrId = ops[0];
			auto sourcePtrId = ops[1];

			auto &targetPointedType =
			    compiler.get_type(compiler.get_type(type_of_temporary_id(targetPtrId)).parent_type);
			auto &sourcePointedType =
			    compiler.get_type(compiler.get_type(type_of_temporary_id(sourcePtrId)).parent_type);
			auto sizeOfPointed1 = size_of(sourcePointedType);
			auto sizeOfPointed2 = size_of(targetPointedType);

			if (sizeOfPointed1 == 0 || sizeOfPointed1 != sizeOfPointed2)
			{
				SPIRV_CROSS_THROW("OpCopyMemory members must be same type, with valid size");
			}

			auto targetPtrPotentials = all_potential_stack_pointer_destinations(targetPtrId, Block.self, Precessor);
			auto sourcePtrPotentials = all_potential_stack_pointer_destinations(sourcePtrId, Block.self, Precessor);

			if (targetPtrPotentials.size() == 1 && sourcePtrPotentials.size() == 1)
			{
				// Trivial case - 2 pointers which can only ever hold static values.
				simulate_memcopy(sourcePtrPotentials.front(), targetPtrPotentials.front(), sizeOfPointed1, Block.self,
				                 Precessor);
			}
			else
			{
				SPIRV_CROSS_THROW("NYI");
			}
			break;
		}
		case spv::OpCopyMemorySized:
		{
			auto targetPtrId = ops[0];
			auto sourcePtrId = ops[1];
			auto sizeId = ops[2];

			auto &targetPointedType =
			    compiler.get_type(compiler.get_type(type_of_temporary_id(targetPtrId)).parent_type);
			auto &sourcePointedType =
			    compiler.get_type(compiler.get_type(type_of_temporary_id(sourcePtrId)).parent_type);

			auto targetPtrPotentials = all_potential_stack_pointer_destinations(targetPtrId, Block.self, Precessor);
			auto sourcePtrPotentials = all_potential_stack_pointer_destinations(sourcePtrId, Block.self, Precessor);

			if (targetPtrPotentials.size() == 1 && sourcePtrPotentials.size() == 1)
			{
				// Simpler case - 2 pointers which can only ever hold static values. Now the only problem
				// is the size, which need not be constant.

				auto validSizes = all_potential_integer_values(sizeId, Block.self);

				if (validSizes.empty())
				{
					validSizes = all_potential_sensible_mem_copy_sizes(sizeId, sourcePtrPotentials.front(),
					                                                   targetPtrPotentials.front(), Block.self);
				}

				if (validSizes.empty())
				{
					SPIRV_CROSS_THROW(
					    "Can't decode OpCopyMemorySized, as the number of bytes to copy couldn't be deduced. ");
				}

				if (validSizes.size() == 1)
				{
					// Sweet, constant, case.
					simulate_memcopy(sourcePtrPotentials.front(), targetPtrPotentials.front(), validSizes.front(),
					                 Block.self, Precessor);
				}
				else
				{
					// We have to put conditional writes.
					SPIRV_CROSS_THROW("NYI");
				}
			}
			else
			{
				// We have variable address we could write to.
				SPIRV_CROSS_THROW("NYI");
			}
			break;
		}
		case spv::OpStore:
		{
			auto targetPtrId = ops[0];
			auto dataId = ops[1];

			auto &targetPointedType =
			    compiler.get_type(compiler.get_type(type_of_temporary_id(targetPtrId)).parent_type);

			auto &sourceType = compiler.get_type(type_of_temporary_id(dataId));

			if (&targetPointedType != &sourceType)
			{
				SPIRV_CROSS_THROW("OpStore pointer type must match data type (missing OpBitCast?)");
			}

			auto targetPtrPotentials = all_potential_stack_pointer_destinations(targetPtrId);

			if (targetPtrPotentials.empty())
			{
				SPIRV_CROSS_THROW(
				    "OpStore failure - We have no idea where we're writing (use set_ptr_hint... functions)");
			}
			else if (targetPtrPotentials.size() == 1)
			{
				// Trivial case - pointer which can point to one place.
				simulate_store(dataId, targetPtrPotentials.front(), size_of(sourceType), Block.self, Precessor);
			}
			else
			{
				// We have multiple write locations, lets look at control flow analysis.
				targetPtrPotentials = all_potential_stack_pointer_destinations(targetPtrId, Block.self, Precessor);

				// We have multiple write locations - this needs an "if (a) b = c;
				// else if (d) e = c; else f = c;" style monstrosity.
				SPIRV_CROSS_THROW("NYI)");
			}
			break;
		}

		case spv::OpLoad:
		{
			// So now we know all id's that may end up stored in memory that we
			// control / know about. Lets see if we can join this load back to the
			// store(s) that made it (with merging, splitting casting, etc glue). The
			// end result (hopefully) is a merging of (two or more) branches on our
			// expression tree, removing all that nasty pointer stuff.

			auto resultType = ops[0];
			auto resultId = ops[1];
			auto &expressionTreeResult = ir.ids[resultId];
			auto ptr = ops[2];

			auto &pointedType = compiler.get_type(compiler.get_type(type_of_temporary_id(ptr)).parent_type);

			auto &resultTypeInfo = compiler.get_type(resultType);
			auto sizeOf = size_of(resultTypeInfo);

			if (&pointedType != &resultTypeInfo)
			{
				SPIRV_CROSS_THROW("OpLoad pointer type must match data type (missing OpBitCast?)");
			}

			auto sourcePtrPotentials = all_potential_stack_pointer_destinations(ptr);

			if (sourcePtrPotentials.size() == 1)
			{
				// Simple case - pointer which can point to one place.

				auto block = try_aligned_uncasted_memory_access(resultTypeInfo, sourcePtrPotentials.front(), sizeOf);

				if (block)
				{
					// Nothing special needed.
					id_data[resultId] = block->value;
				}
				else
				{
					auto blocks = try_aligned_access_memory(resultTypeInfo, sourcePtrPotentials.front(), sizeOf);

					if (blocks.empty())
					{
						// Read isn't aligned - TODO
						SPIRV_CROSS_THROW("NYI");
					}
					else
					{
						if (blocks.size() == 1)
						{
							// This is a cast
							auto cast = std::make_shared<DataCast>();
							cast->input = block->value;
							cast->input_type = block->type;
							cast->output_type = resultType;
							id_data[resultId] = cast;
						}
						else
						{
							// This is a recomposition
							auto values = std::make_shared<DataComposition>();
							for (auto b : blocks)
							{
								values->inputs.push_back(b->value);
							}
							values->output_type = resultType;
							id_data[resultId] = values;
						}
					}
				}
			}
			else
			{
				SPIRV_CROSS_THROW("NYI - multi pointer load");
			}
			break;
		}
		}
	}

	return {};
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

bool spirv_cross::MemoryAnalyser::simulate_memcopy(uint32_t SourceAddress, uint32_t DestinationAddress,
                                                   uint32_t SizeInBytes, uint32_t ActiveBlock, uint32_t PrecedingBlock)
{
	auto offset = 0u;

	while (offset < SizeInBytes)
	{
		auto sourceNode = find_memory_leaf_allocation(SourceAddress + offset);
		auto destNode = find_memory_leaf_allocation(DestinationAddress + offset);

		if (sourceNode->address != SourceAddress + offset)
		{
			SPIRV_CROSS_THROW("Misaligned read");
		}

		if (destNode->address != SourceAddress + offset)
		{
			SPIRV_CROSS_THROW("Misaligned write");
		}

		if (sourceNode->type != destNode->type)
		{
			SPIRV_CROSS_THROW("NYI - casting mem copy");
		}

		if (sourceNode->size_in_bytes != destNode->size_in_bytes)
		{
			SPIRV_CROSS_THROW("NYI - subdivide memory tree");
		}

		if (destNode->value)
		{
			// We're writing over memory. (We can't destNode->value = sourceNode->value as the old value
			// may have been read elsewere)
			SPIRV_CROSS_THROW("NYI - phi memory trees");
		}
		else
		{
			// We're the first time a value has been written here.
			destNode->value = sourceNode->value;
		}

		offset += sourceNode->size_in_bytes;
	}

	return false;
}

bool spirv_cross::MemoryAnalyser::simulate_store(uint32_t SourceId, uint32_t DestinationAddress, uint32_t SizeInBytes,
                                                 uint32_t ActiveBlock, uint32_t PrecedingBlock)
{
	auto &data = id_data[SourceId];

	if (!data)
	{
		auto maybeExpression = compiler.maybe_get<SPIRExpression>(SourceId);
		auto maybeConstant = compiler.maybe_get<SPIRConstant>(SourceId);
		auto maybeConstantOp = compiler.maybe_get<SPIRConstantOp>(SourceId);

		if (maybeExpression)
		{
			struct DataExpression : Data
			{
				SPIRExpression *value = nullptr;
				Resolution resolved(UInt32TOptional CurrentBlock, UInt32TOptional PrecedingBlock) const override
				{
					return ExistingId;
				}

				uint32_t id(UInt32TOptional CurrentBlock, UInt32TOptional PrecedingBlock) const override
				{
					return value->self;
				}

				bool needs_backing_variable() const override
				{
					return false;
				}
			};

			auto de = std::make_shared<DataExpression>();
			de->value = maybeExpression;
			data = de;
		}
		else if (maybeConstant || maybeConstantOp)
		{
			auto dc = std::make_shared<DataConstant>();

			if (maybeConstant)
			{
				auto constantType = compiler.get_type(maybeConstant->constant_type);
				dc->isScalar = (constantType.parent_type == 0);
				dc->valueConstant = maybeConstant;
				dc->type = maybeConstant->constant_type;
			}
			else
			{
				dc->valueConstExpr = maybeConstantOp;
				dc->type = maybeConstantOp->basetype;
			}

			data = dc;
		}
		else
		{
			SPIRV_CROSS_THROW("NYI");
		}
	}

	auto destNode = find_memory_location(DestinationAddress, SizeInBytes);

	if (destNode->address != DestinationAddress)
	{
		SPIRV_CROSS_THROW("Misaligned write handler NYI");
		// data = MisAlignedHandler(data)
	}

	if (destNode->value)
	{
		// We're writing over memory. (We can't destNode->value = sourceNode->value as the old value
		// may have been written elsewere)
		SPIRV_CROSS_THROW("NYI - phi memory trees ");
	}
	else
	{
		// We're the first time a value has been written here.
		destNode->value = data;
	}

	if (destNode->has_child)
	{
		// We've written to a level in memory which has members, we need to update all the children to
		// refer to the new value.

		for (auto &child : destNode->children)
		{
			auto childData = std::make_shared<DataAccessChain>();
			childData->parent_value = data;
			childData->access_chain = { child.local_access_chain.begin() + destNode->local_access_chain.size(),
				                        child.local_access_chain.end() };

			childData->relative_pointer = int32_t(child.address - destNode->address);

			if (child.value)
			{
				// We're writing over memory.
				SPIRV_CROSS_THROW("NYI - phi memory trees ");
			}
			else
			{
				// We're the first time a value has been written here.
				child.value = childData;
			}
		}
	}

	return false;
}
