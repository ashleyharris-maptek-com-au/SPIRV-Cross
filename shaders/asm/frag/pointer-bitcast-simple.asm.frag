; SPIR-V
; Version: 1.3
; Generator: Khronos; 0
; Bound: 28
; Schema: 0
               OpCapability Shader
               OpCapability Addresses
               OpMemoryModel Physical32 GLSL450
               OpEntryPoint Fragment %main "main" %pFragColour
               OpExecutionMode %main OriginLowerLeft
               OpName %Tvoid "Tvoid"
               OpName %Tf_Tvoid "Tf_Tvoid"
               OpName %Tf32 "Tf32"
               OpName %Ti32u "Ti32u"
               OpName %Tarr8Tf32 "Tarr8Tf32"
               OpName %TppTarr8Tf32 "TppTarr8Tf32"
               OpName %TppTf32 "TppTf32"
               OpName %Tvec4 "Tvec4"
               OpName %TppTvec4 "TppTvec4"
               OpName %TpoTvec4 "TpoTvec4"
               OpName %pFragColour "pFragColour"
               OpName %Lmain "Lmain"
               OpName %main "main"
               OpDecorate %pFragColour Location 0
      %Tvoid = OpTypeVoid
   %Tf_Tvoid = OpTypeFunction %Tvoid
       %Tf32 = OpTypeFloat 32
      %Ti32u = OpTypeInt 32 0
    %Ti32u_8 = OpConstant %Ti32u 8
  %Tarr8Tf32 = OpTypeArray %Tf32 %Ti32u_8
%TppTarr8Tf32 = OpTypePointer Private %Tarr8Tf32
    %TppTf32 = OpTypePointer Private %Tf32
    %Ti32u_0 = OpConstant %Ti32u 0
    %Ti32u_1 = OpConstant %Ti32u 1
    %Ti32u_2 = OpConstant %Ti32u 2
    %Ti32u_3 = OpConstant %Ti32u 3
     %Tf32_1 = OpConstant %Tf32 1
     %Tf32_0 = OpConstant %Tf32 0
      %Tvec4 = OpTypeVector %Tf32 4
   %TppTvec4 = OpTypePointer Private %Tvec4
   %TpoTvec4 = OpTypePointer Output %Tvec4
%pFragColour = OpVariable %TpoTvec4 Output
       %main = OpFunction %Tvoid None %Tf_Tvoid
      %Lmain = OpLabel
         %10 = OpVariable %TppTarr8Tf32 Function
         %12 = OpInBoundsAccessChain %TppTf32 %10 %Ti32u_0
         %14 = OpInBoundsAccessChain %TppTf32 %10 %Ti32u_1
         %16 = OpInBoundsAccessChain %TppTf32 %10 %Ti32u_2
         %18 = OpInBoundsAccessChain %TppTf32 %10 %Ti32u_3
               OpStore %12 %Tf32_0
               OpStore %14 %Tf32_1
               OpStore %16 %Tf32_1
               OpStore %18 %Tf32_1
         %24 = OpBitcast %TppTvec4 %10
         %25 = OpLoad %Tvec4 %24
               OpStore %pFragColour %25
               OpReturn
               OpFunctionEnd
