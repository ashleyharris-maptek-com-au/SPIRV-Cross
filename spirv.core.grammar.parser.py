# 
# Copyright 2018-2019 Ashley Harris (Maptek Australia Pty Ltd)
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

import json, subprocess
from pprint import pprint

with open('spirv.core.grammar.json') as f:
    data = json.load(f)

with open('spirv.hpp') as f:
    referenceCpp = f.read()
    
cppCodeHasResult = """
// Returns true if an operation has a return value.
constexpr bool has_result(spv::Op op)
{
  switch(op)
  {
""";

cppCodeIsInputId = """
// Returns true if an operation's arg'th parameter is an id.
// (If not, it may be a litteral, an enum, or some other thing).
constexpr bool is_input_id(spv::Op op, uint32_t arg)
{
  switch(op)
  {
""";
    
for instruction in data["instructions"]:
  opName = instruction["opname"]
  
  if (opName not in referenceCpp): 
    print opName + " isn't in spirv.hpp - skipping"
    continue
  
  opCount = 0
  if ("operands" in instruction): opCount = len(instruction["operands"])
  
  hasResultType = False
  hasResultId = False
  
  isInputId = []
  
  extraIdsAfter = None
  
  for i in range(0, opCount):
    op = instruction["operands"][i]
    
    if op["kind"] == "IdRef":
      isInputId.append("true")
    else:
      isInputId.append("false")
    
    if op["kind"] == "IdResultType": hasResultType = True
    if op["kind"] == "IdResult": hasResultId = True
    
    if "quatifier" in op and op["quatifier"] == '*':
      if op["kind"] == "IdRef":
        extraIdsAfter = i
        
  if hasResultId and hasResultType:
    cppCodeHasResult += "case spv::" + opName + ":\n"
    
  if (opCount == 0):
    cppCodeIsInputId += "case spv::" + opName + ": return false;\n"
  
  else:
    cppCodeIsInputId += ("case spv::" + opName + 
      ":\n{\n  auto is_id_ref = {" + ", ".join(isInputId) + "};\n")
    
    if (extraIdsAfter is not None):
      cppCodeIsInputId += "  if (arg >= " + str(extraIdsAfter) + ") return true;\n"
    else:
      cppCodeIsInputId += "  if (arg >= " + str(opCount) + ") return false;\n"
    
    cppCodeIsInputId += "  return is_id_ref.begin()[arg];\n}\n";
    
    
cppCodeHasResult += """  return true;
default:
  return false;
  }
}


""";

cppCodeIsInputId += """
default: return false;
  }
}


""";

cmd = ["clang-format" , '-style=file', '-assume-filename=spirv_arguments.hpp']

clangFormatInput = """
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
 
 // Some useful functions to analyse an instruction.
 //
 // This file is autogenerated from spirv.core.grammar.parser.py from 
 // spirv.core.grammar.json.
 

#ifndef SPIRV_ARGUMENTS_HPP
#define SPIRV_ARGUMENTS_HPP

#include "spirv.hpp"

#include <initializer_list>
#include <cstdint>

namespace spirv_cross
{
  """ + cppCodeHasResult + cppCodeIsInputId + """
}

#endif
"""

try:
  task = subprocess.Popen(cmd, stderr=subprocess.STDOUT, stdout = subprocess.PIPE, stdin = subprocess.PIPE)
  (output, error) = task.communicate(clangFormatInput)
except:
  print "Couldn't run clang-format, so spirv_arguments.hpp is unformatted"
  output = clangFormatInput
  
f = open("spirv_arguments.hpp", "w")
f.write(output)
f.close()

