��
��
B
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8��
�
wide_deep/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*)
shared_namewide_deep/dense_4/kernel
�
,wide_deep/dense_4/kernel/Read/ReadVariableOpReadVariableOpwide_deep/dense_4/kernel*
_output_shapes

:@*
dtype0
�
wide_deep/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namewide_deep/dense_4/bias
}
*wide_deep/dense_4/bias/Read/ReadVariableOpReadVariableOpwide_deep/dense_4/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
�
wide_deep/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ͷ*/
shared_name wide_deep/embedding/embeddings
�
2wide_deep/embedding/embeddings/Read/ReadVariableOpReadVariableOpwide_deep/embedding/embeddings* 
_output_shapes
:
Ͷ*
dtype0
�
 wide_deep/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*1
shared_name" wide_deep/embedding_1/embeddings
�
4wide_deep/embedding_1/embeddings/Read/ReadVariableOpReadVariableOp wide_deep/embedding_1/embeddings* 
_output_shapes
:
��*
dtype0
�
 wide_deep/embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*1
shared_name" wide_deep/embedding_2/embeddings
�
4wide_deep/embedding_2/embeddings/Read/ReadVariableOpReadVariableOp wide_deep/embedding_2/embeddings*
_output_shapes
:	�*
dtype0
�
 wide_deep/embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*1
shared_name" wide_deep/embedding_3/embeddings
�
4wide_deep/embedding_3/embeddings/Read/ReadVariableOpReadVariableOp wide_deep/embedding_3/embeddings*
_output_shapes

:*
dtype0
�
 wide_deep/embedding_4/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*1
shared_name" wide_deep/embedding_4/embeddings
�
4wide_deep/embedding_4/embeddings/Read/ReadVariableOpReadVariableOp wide_deep/embedding_4/embeddings* 
_output_shapes
:
��*
dtype0
�
wide_deep/dnn/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	1�*+
shared_namewide_deep/dnn/dense/kernel
�
.wide_deep/dnn/dense/kernel/Read/ReadVariableOpReadVariableOpwide_deep/dnn/dense/kernel*
_output_shapes
:	1�*
dtype0
�
wide_deep/dnn/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*)
shared_namewide_deep/dnn/dense/bias
�
,wide_deep/dnn/dense/bias/Read/ReadVariableOpReadVariableOpwide_deep/dnn/dense/bias*
_output_shapes	
:�*
dtype0
�
wide_deep/dnn/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*-
shared_namewide_deep/dnn/dense_1/kernel
�
0wide_deep/dnn/dense_1/kernel/Read/ReadVariableOpReadVariableOpwide_deep/dnn/dense_1/kernel* 
_output_shapes
:
��*
dtype0
�
wide_deep/dnn/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_namewide_deep/dnn/dense_1/bias
�
.wide_deep/dnn/dense_1/bias/Read/ReadVariableOpReadVariableOpwide_deep/dnn/dense_1/bias*
_output_shapes	
:�*
dtype0
�
wide_deep/dnn/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*-
shared_namewide_deep/dnn/dense_2/kernel
�
0wide_deep/dnn/dense_2/kernel/Read/ReadVariableOpReadVariableOpwide_deep/dnn/dense_2/kernel*
_output_shapes
:	�@*
dtype0
�
wide_deep/dnn/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namewide_deep/dnn/dense_2/bias
�
.wide_deep/dnn/dense_2/bias/Read/ReadVariableOpReadVariableOpwide_deep/dnn/dense_2/bias*
_output_shapes
:@*
dtype0
�
wide_deep/linear/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*0
shared_name!wide_deep/linear/dense_3/kernel
�
3wide_deep/linear/dense_3/kernel/Read/ReadVariableOpReadVariableOpwide_deep/linear/dense_3/kernel*
_output_shapes

:	*
dtype0
�
wide_deep/linear/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_namewide_deep/linear/dense_3/bias
�
1wide_deep/linear/dense_3/bias/Read/ReadVariableOpReadVariableOpwide_deep/linear/dense_3/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
u
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_positives
n
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes	
:�*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:�*
dtype0
w
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_positives
p
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes	
:�*
dtype0
w
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:�* 
shared_namefalse_negatives
p
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes	
:�*
dtype0
�
Adam/wide_deep/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!Adam/wide_deep/dense_4/kernel/m
�
3Adam/wide_deep/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/wide_deep/dense_4/kernel/m*
_output_shapes

:@*
dtype0
�
Adam/wide_deep/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/wide_deep/dense_4/bias/m
�
1Adam/wide_deep/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/wide_deep/dense_4/bias/m*
_output_shapes
:*
dtype0
�
%Adam/wide_deep/embedding/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ͷ*6
shared_name'%Adam/wide_deep/embedding/embeddings/m
�
9Adam/wide_deep/embedding/embeddings/m/Read/ReadVariableOpReadVariableOp%Adam/wide_deep/embedding/embeddings/m* 
_output_shapes
:
Ͷ*
dtype0
�
'Adam/wide_deep/embedding_1/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*8
shared_name)'Adam/wide_deep/embedding_1/embeddings/m
�
;Adam/wide_deep/embedding_1/embeddings/m/Read/ReadVariableOpReadVariableOp'Adam/wide_deep/embedding_1/embeddings/m* 
_output_shapes
:
��*
dtype0
�
'Adam/wide_deep/embedding_2/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*8
shared_name)'Adam/wide_deep/embedding_2/embeddings/m
�
;Adam/wide_deep/embedding_2/embeddings/m/Read/ReadVariableOpReadVariableOp'Adam/wide_deep/embedding_2/embeddings/m*
_output_shapes
:	�*
dtype0
�
'Adam/wide_deep/embedding_3/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'Adam/wide_deep/embedding_3/embeddings/m
�
;Adam/wide_deep/embedding_3/embeddings/m/Read/ReadVariableOpReadVariableOp'Adam/wide_deep/embedding_3/embeddings/m*
_output_shapes

:*
dtype0
�
'Adam/wide_deep/embedding_4/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*8
shared_name)'Adam/wide_deep/embedding_4/embeddings/m
�
;Adam/wide_deep/embedding_4/embeddings/m/Read/ReadVariableOpReadVariableOp'Adam/wide_deep/embedding_4/embeddings/m* 
_output_shapes
:
��*
dtype0
�
!Adam/wide_deep/dnn/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	1�*2
shared_name#!Adam/wide_deep/dnn/dense/kernel/m
�
5Adam/wide_deep/dnn/dense/kernel/m/Read/ReadVariableOpReadVariableOp!Adam/wide_deep/dnn/dense/kernel/m*
_output_shapes
:	1�*
dtype0
�
Adam/wide_deep/dnn/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!Adam/wide_deep/dnn/dense/bias/m
�
3Adam/wide_deep/dnn/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/wide_deep/dnn/dense/bias/m*
_output_shapes	
:�*
dtype0
�
#Adam/wide_deep/dnn/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#Adam/wide_deep/dnn/dense_1/kernel/m
�
7Adam/wide_deep/dnn/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/wide_deep/dnn/dense_1/kernel/m* 
_output_shapes
:
��*
dtype0
�
!Adam/wide_deep/dnn/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/wide_deep/dnn/dense_1/bias/m
�
5Adam/wide_deep/dnn/dense_1/bias/m/Read/ReadVariableOpReadVariableOp!Adam/wide_deep/dnn/dense_1/bias/m*
_output_shapes	
:�*
dtype0
�
#Adam/wide_deep/dnn/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*4
shared_name%#Adam/wide_deep/dnn/dense_2/kernel/m
�
7Adam/wide_deep/dnn/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/wide_deep/dnn/dense_2/kernel/m*
_output_shapes
:	�@*
dtype0
�
!Adam/wide_deep/dnn/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/wide_deep/dnn/dense_2/bias/m
�
5Adam/wide_deep/dnn/dense_2/bias/m/Read/ReadVariableOpReadVariableOp!Adam/wide_deep/dnn/dense_2/bias/m*
_output_shapes
:@*
dtype0
�
&Adam/wide_deep/linear/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*7
shared_name(&Adam/wide_deep/linear/dense_3/kernel/m
�
:Adam/wide_deep/linear/dense_3/kernel/m/Read/ReadVariableOpReadVariableOp&Adam/wide_deep/linear/dense_3/kernel/m*
_output_shapes

:	*
dtype0
�
$Adam/wide_deep/linear/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/wide_deep/linear/dense_3/bias/m
�
8Adam/wide_deep/linear/dense_3/bias/m/Read/ReadVariableOpReadVariableOp$Adam/wide_deep/linear/dense_3/bias/m*
_output_shapes
:*
dtype0
�
Adam/wide_deep/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!Adam/wide_deep/dense_4/kernel/v
�
3Adam/wide_deep/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/wide_deep/dense_4/kernel/v*
_output_shapes

:@*
dtype0
�
Adam/wide_deep/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/wide_deep/dense_4/bias/v
�
1Adam/wide_deep/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/wide_deep/dense_4/bias/v*
_output_shapes
:*
dtype0
�
%Adam/wide_deep/embedding/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
Ͷ*6
shared_name'%Adam/wide_deep/embedding/embeddings/v
�
9Adam/wide_deep/embedding/embeddings/v/Read/ReadVariableOpReadVariableOp%Adam/wide_deep/embedding/embeddings/v* 
_output_shapes
:
Ͷ*
dtype0
�
'Adam/wide_deep/embedding_1/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*8
shared_name)'Adam/wide_deep/embedding_1/embeddings/v
�
;Adam/wide_deep/embedding_1/embeddings/v/Read/ReadVariableOpReadVariableOp'Adam/wide_deep/embedding_1/embeddings/v* 
_output_shapes
:
��*
dtype0
�
'Adam/wide_deep/embedding_2/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�*8
shared_name)'Adam/wide_deep/embedding_2/embeddings/v
�
;Adam/wide_deep/embedding_2/embeddings/v/Read/ReadVariableOpReadVariableOp'Adam/wide_deep/embedding_2/embeddings/v*
_output_shapes
:	�*
dtype0
�
'Adam/wide_deep/embedding_3/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*8
shared_name)'Adam/wide_deep/embedding_3/embeddings/v
�
;Adam/wide_deep/embedding_3/embeddings/v/Read/ReadVariableOpReadVariableOp'Adam/wide_deep/embedding_3/embeddings/v*
_output_shapes

:*
dtype0
�
'Adam/wide_deep/embedding_4/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*8
shared_name)'Adam/wide_deep/embedding_4/embeddings/v
�
;Adam/wide_deep/embedding_4/embeddings/v/Read/ReadVariableOpReadVariableOp'Adam/wide_deep/embedding_4/embeddings/v* 
_output_shapes
:
��*
dtype0
�
!Adam/wide_deep/dnn/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	1�*2
shared_name#!Adam/wide_deep/dnn/dense/kernel/v
�
5Adam/wide_deep/dnn/dense/kernel/v/Read/ReadVariableOpReadVariableOp!Adam/wide_deep/dnn/dense/kernel/v*
_output_shapes
:	1�*
dtype0
�
Adam/wide_deep/dnn/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*0
shared_name!Adam/wide_deep/dnn/dense/bias/v
�
3Adam/wide_deep/dnn/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/wide_deep/dnn/dense/bias/v*
_output_shapes	
:�*
dtype0
�
#Adam/wide_deep/dnn/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*4
shared_name%#Adam/wide_deep/dnn/dense_1/kernel/v
�
7Adam/wide_deep/dnn/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/wide_deep/dnn/dense_1/kernel/v* 
_output_shapes
:
��*
dtype0
�
!Adam/wide_deep/dnn/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/wide_deep/dnn/dense_1/bias/v
�
5Adam/wide_deep/dnn/dense_1/bias/v/Read/ReadVariableOpReadVariableOp!Adam/wide_deep/dnn/dense_1/bias/v*
_output_shapes	
:�*
dtype0
�
#Adam/wide_deep/dnn/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*4
shared_name%#Adam/wide_deep/dnn/dense_2/kernel/v
�
7Adam/wide_deep/dnn/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/wide_deep/dnn/dense_2/kernel/v*
_output_shapes
:	�@*
dtype0
�
!Adam/wide_deep/dnn/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!Adam/wide_deep/dnn/dense_2/bias/v
�
5Adam/wide_deep/dnn/dense_2/bias/v/Read/ReadVariableOpReadVariableOp!Adam/wide_deep/dnn/dense_2/bias/v*
_output_shapes
:@*
dtype0
�
&Adam/wide_deep/linear/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*7
shared_name(&Adam/wide_deep/linear/dense_3/kernel/v
�
:Adam/wide_deep/linear/dense_3/kernel/v/Read/ReadVariableOpReadVariableOp&Adam/wide_deep/linear/dense_3/kernel/v*
_output_shapes

:	*
dtype0
�
$Adam/wide_deep/linear/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/wide_deep/linear/dense_3/bias/v
�
8Adam/wide_deep/linear/dense_3/bias/v/Read/ReadVariableOpReadVariableOp$Adam/wide_deep/linear/dense_3/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�]
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�]
value�]B�] B�]
�
dense_feature_columns
sparse_feature_columns
embed_layers
dnn_network

linear
final_dense
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
?
0
1
2
3
4
5
6
7
8
#
0
1
2
3
4
A
embed_0
embed_1
embed_2
embed_3
embed_4
p
 dnn_network
!dropout
"regularization_losses
#	variables
$trainable_variables
%	keras_api
]
	&dense
'regularization_losses
(	variables
)trainable_variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
�
1iter

2beta_1

3beta_2
	4decay
5learning_rate+m�,m�6m�7m�8m�9m�:m�;m�<m�=m�>m�?m�@m�Am�Bm�+v�,v�6v�7v�8v�9v�:v�;v�<v�=v�>v�?v�@v�Av�Bv�
 
n
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11
B12
+13
,14
n
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11
B12
+13
,14
�
regularization_losses
Cnon_trainable_variables
		variables

Dlayers
Elayer_metrics

trainable_variables
Flayer_regularization_losses
Gmetrics
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
b
6
embeddings
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
b
7
embeddings
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
b
8
embeddings
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
b
9
embeddings
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
b
:
embeddings
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api

\0
]1
^2
R
_regularization_losses
`	variables
atrainable_variables
b	keras_api
 
*
;0
<1
=2
>3
?4
@5
*
;0
<1
=2
>3
?4
@5
�
"regularization_losses
cnon_trainable_variables
#	variables
dlayer_metrics

elayers
$trainable_variables
flayer_regularization_losses
gmetrics
h

Akernel
Bbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
 

A0
B1

A0
B1
�
'regularization_losses
lnon_trainable_variables
(	variables
mlayer_metrics

nlayers
)trainable_variables
olayer_regularization_losses
pmetrics
[Y
VARIABLE_VALUEwide_deep/dense_4/kernel-final_dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEwide_deep/dense_4/bias+final_dense/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
�
-regularization_losses
qnon_trainable_variables
.	variables
rlayer_metrics

slayers
/trainable_variables
tlayer_regularization_losses
umetrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEwide_deep/embedding/embeddings&variables/0/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE wide_deep/embedding_1/embeddings&variables/1/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE wide_deep/embedding_2/embeddings&variables/2/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE wide_deep/embedding_3/embeddings&variables/3/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE wide_deep/embedding_4/embeddings&variables/4/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEwide_deep/dnn/dense/kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEwide_deep/dnn/dense/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEwide_deep/dnn/dense_1/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEwide_deep/dnn/dense_1/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEwide_deep/dnn/dense_2/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEwide_deep/dnn/dense_2/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEwide_deep/linear/dense_3/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEwide_deep/linear/dense_3/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE
 
8
0
1
2
3
4
5
6
7
 
 

v0
w1
 

60

60
�
Hregularization_losses
xnon_trainable_variables
I	variables
ylayer_metrics

zlayers
Jtrainable_variables
{layer_regularization_losses
|metrics
 

70

70
�
Lregularization_losses
}non_trainable_variables
M	variables
~layer_metrics

layers
Ntrainable_variables
 �layer_regularization_losses
�metrics
 

80

80
�
Pregularization_losses
�non_trainable_variables
Q	variables
�layer_metrics
�layers
Rtrainable_variables
 �layer_regularization_losses
�metrics
 

90

90
�
Tregularization_losses
�non_trainable_variables
U	variables
�layer_metrics
�layers
Vtrainable_variables
 �layer_regularization_losses
�metrics
 

:0

:0
�
Xregularization_losses
�non_trainable_variables
Y	variables
�layer_metrics
�layers
Ztrainable_variables
 �layer_regularization_losses
�metrics
l

;kernel
<bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
l

=kernel
>bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
l

?kernel
@bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
 
 
 
�
_regularization_losses
�non_trainable_variables
`	variables
�layer_metrics
�layers
atrainable_variables
 �layer_regularization_losses
�metrics
 
 

\0
]1
^2
!3
 
 
 

A0
B1

A0
B1
�
hregularization_losses
�non_trainable_variables
i	variables
�layer_metrics
�layers
jtrainable_variables
 �layer_regularization_losses
�metrics
 
 

&0
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
v
�true_positives
�true_negatives
�false_positives
�false_negatives
�	variables
�	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

;0
<1

;0
<1
�
�regularization_losses
�non_trainable_variables
�	variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
 

=0
>1

=0
>1
�
�regularization_losses
�non_trainable_variables
�	variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
 

?0
@1

?0
@1
�
�regularization_losses
�non_trainable_variables
�	variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUE
 
�0
�1
�2
�3

�	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
~|
VARIABLE_VALUEAdam/wide_deep/dense_4/kernel/mIfinal_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/wide_deep/dense_4/bias/mGfinal_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%Adam/wide_deep/embedding/embeddings/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/wide_deep/embedding_1/embeddings/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/wide_deep/embedding_2/embeddings/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/wide_deep/embedding_3/embeddings/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/wide_deep/embedding_4/embeddings/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/wide_deep/dnn/dense/kernel/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/wide_deep/dnn/dense/bias/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/wide_deep/dnn/dense_1/kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/wide_deep/dnn/dense_1/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/wide_deep/dnn/dense_2/kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/wide_deep/dnn/dense_2/bias/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE&Adam/wide_deep/linear/dense_3/kernel/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE$Adam/wide_deep/linear/dense_3/bias/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/wide_deep/dense_4/kernel/vIfinal_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/wide_deep/dense_4/bias/vGfinal_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE%Adam/wide_deep/embedding/embeddings/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/wide_deep/embedding_1/embeddings/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/wide_deep/embedding_2/embeddings/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/wide_deep/embedding_3/embeddings/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE'Adam/wide_deep/embedding_4/embeddings/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/wide_deep/dnn/dense/kernel/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/wide_deep/dnn/dense/bias/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/wide_deep/dnn/dense_1/kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!Adam/wide_deep/dnn/dense_1/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE#Adam/wide_deep/dnn/dense_2/kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE!Adam/wide_deep/dnn/dense_2/bias/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUE&Adam/wide_deep/linear/dense_3/kernel/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE$Adam/wide_deep/linear/dense_3/bias/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:���������	*
dtype0*
shape:���������	
z
serving_default_input_2Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2wide_deep/embedding/embeddings wide_deep/embedding_1/embeddings wide_deep/embedding_2/embeddings wide_deep/embedding_3/embeddings wide_deep/embedding_4/embeddingswide_deep/linear/dense_3/kernelwide_deep/linear/dense_3/biaswide_deep/dnn/dense/kernelwide_deep/dnn/dense/biaswide_deep/dnn/dense_1/kernelwide_deep/dnn/dense_1/biaswide_deep/dnn/dense_2/kernelwide_deep/dnn/dense_2/biaswide_deep/dense_4/kernelwide_deep/dense_4/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_3719
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,wide_deep/dense_4/kernel/Read/ReadVariableOp*wide_deep/dense_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp2wide_deep/embedding/embeddings/Read/ReadVariableOp4wide_deep/embedding_1/embeddings/Read/ReadVariableOp4wide_deep/embedding_2/embeddings/Read/ReadVariableOp4wide_deep/embedding_3/embeddings/Read/ReadVariableOp4wide_deep/embedding_4/embeddings/Read/ReadVariableOp.wide_deep/dnn/dense/kernel/Read/ReadVariableOp,wide_deep/dnn/dense/bias/Read/ReadVariableOp0wide_deep/dnn/dense_1/kernel/Read/ReadVariableOp.wide_deep/dnn/dense_1/bias/Read/ReadVariableOp0wide_deep/dnn/dense_2/kernel/Read/ReadVariableOp.wide_deep/dnn/dense_2/bias/Read/ReadVariableOp3wide_deep/linear/dense_3/kernel/Read/ReadVariableOp1wide_deep/linear/dense_3/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp3Adam/wide_deep/dense_4/kernel/m/Read/ReadVariableOp1Adam/wide_deep/dense_4/bias/m/Read/ReadVariableOp9Adam/wide_deep/embedding/embeddings/m/Read/ReadVariableOp;Adam/wide_deep/embedding_1/embeddings/m/Read/ReadVariableOp;Adam/wide_deep/embedding_2/embeddings/m/Read/ReadVariableOp;Adam/wide_deep/embedding_3/embeddings/m/Read/ReadVariableOp;Adam/wide_deep/embedding_4/embeddings/m/Read/ReadVariableOp5Adam/wide_deep/dnn/dense/kernel/m/Read/ReadVariableOp3Adam/wide_deep/dnn/dense/bias/m/Read/ReadVariableOp7Adam/wide_deep/dnn/dense_1/kernel/m/Read/ReadVariableOp5Adam/wide_deep/dnn/dense_1/bias/m/Read/ReadVariableOp7Adam/wide_deep/dnn/dense_2/kernel/m/Read/ReadVariableOp5Adam/wide_deep/dnn/dense_2/bias/m/Read/ReadVariableOp:Adam/wide_deep/linear/dense_3/kernel/m/Read/ReadVariableOp8Adam/wide_deep/linear/dense_3/bias/m/Read/ReadVariableOp3Adam/wide_deep/dense_4/kernel/v/Read/ReadVariableOp1Adam/wide_deep/dense_4/bias/v/Read/ReadVariableOp9Adam/wide_deep/embedding/embeddings/v/Read/ReadVariableOp;Adam/wide_deep/embedding_1/embeddings/v/Read/ReadVariableOp;Adam/wide_deep/embedding_2/embeddings/v/Read/ReadVariableOp;Adam/wide_deep/embedding_3/embeddings/v/Read/ReadVariableOp;Adam/wide_deep/embedding_4/embeddings/v/Read/ReadVariableOp5Adam/wide_deep/dnn/dense/kernel/v/Read/ReadVariableOp3Adam/wide_deep/dnn/dense/bias/v/Read/ReadVariableOp7Adam/wide_deep/dnn/dense_1/kernel/v/Read/ReadVariableOp5Adam/wide_deep/dnn/dense_1/bias/v/Read/ReadVariableOp7Adam/wide_deep/dnn/dense_2/kernel/v/Read/ReadVariableOp5Adam/wide_deep/dnn/dense_2/bias/v/Read/ReadVariableOp:Adam/wide_deep/linear/dense_3/kernel/v/Read/ReadVariableOp8Adam/wide_deep/linear/dense_3/bias/v/Read/ReadVariableOpConst*E
Tin>
<2:	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_4583
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamewide_deep/dense_4/kernelwide_deep/dense_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratewide_deep/embedding/embeddings wide_deep/embedding_1/embeddings wide_deep/embedding_2/embeddings wide_deep/embedding_3/embeddings wide_deep/embedding_4/embeddingswide_deep/dnn/dense/kernelwide_deep/dnn/dense/biaswide_deep/dnn/dense_1/kernelwide_deep/dnn/dense_1/biaswide_deep/dnn/dense_2/kernelwide_deep/dnn/dense_2/biaswide_deep/linear/dense_3/kernelwide_deep/linear/dense_3/biastotalcounttrue_positivestrue_negativesfalse_positivesfalse_negativesAdam/wide_deep/dense_4/kernel/mAdam/wide_deep/dense_4/bias/m%Adam/wide_deep/embedding/embeddings/m'Adam/wide_deep/embedding_1/embeddings/m'Adam/wide_deep/embedding_2/embeddings/m'Adam/wide_deep/embedding_3/embeddings/m'Adam/wide_deep/embedding_4/embeddings/m!Adam/wide_deep/dnn/dense/kernel/mAdam/wide_deep/dnn/dense/bias/m#Adam/wide_deep/dnn/dense_1/kernel/m!Adam/wide_deep/dnn/dense_1/bias/m#Adam/wide_deep/dnn/dense_2/kernel/m!Adam/wide_deep/dnn/dense_2/bias/m&Adam/wide_deep/linear/dense_3/kernel/m$Adam/wide_deep/linear/dense_3/bias/mAdam/wide_deep/dense_4/kernel/vAdam/wide_deep/dense_4/bias/v%Adam/wide_deep/embedding/embeddings/v'Adam/wide_deep/embedding_1/embeddings/v'Adam/wide_deep/embedding_2/embeddings/v'Adam/wide_deep/embedding_3/embeddings/v'Adam/wide_deep/embedding_4/embeddings/v!Adam/wide_deep/dnn/dense/kernel/vAdam/wide_deep/dnn/dense/bias/v#Adam/wide_deep/dnn/dense_1/kernel/v!Adam/wide_deep/dnn/dense_1/bias/v#Adam/wide_deep/dnn/dense_2/kernel/v!Adam/wide_deep/dnn/dense_2/bias/v&Adam/wide_deep/linear/dense_3/kernel/v$Adam/wide_deep/linear/dense_3/bias/v*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_4761��
�
�
E__inference_embedding_1_layer_call_and_return_conditional_losses_2936

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02!
embedding_lookup/ReadVariableOp�
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis�
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:���������2
embedding_lookup/Identity�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_1/embeddings/Regularizer/SquareSquareJwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_1/embeddings/Regularizer/Square�
2wide_deep/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_1/embeddings/Regularizer/Const�
0wide_deep/embedding_1/embeddings/Regularizer/SumSum7wide_deep/embedding_1/embeddings/Regularizer/Square:y:0;wide_deep/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/Sum�
2wide_deep/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_1/embeddings/Regularizer/mul/x�
0wide_deep/embedding_1/embeddings/Regularizer/mulMul;wide_deep/embedding_1/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/mul�
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOpC^wide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp2�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_embedding_4_layer_call_and_return_conditional_losses_3029

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02!
embedding_lookup/ReadVariableOp�
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis�
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:���������2
embedding_lookup/Identity�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_4/embeddings/Regularizer/SquareSquareJwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_4/embeddings/Regularizer/Square�
2wide_deep/embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_4/embeddings/Regularizer/Const�
0wide_deep/embedding_4/embeddings/Regularizer/SumSum7wide_deep/embedding_4/embeddings/Regularizer/Square:y:0;wide_deep/embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/Sum�
2wide_deep/embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_4/embeddings/Regularizer/mul/x�
0wide_deep/embedding_4/embeddings/Regularizer/mulMul;wide_deep/embedding_4/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/mul�
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOpC^wide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp2�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�&
�
=__inference_dnn_layer_call_and_return_conditional_losses_4079

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	1�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_1/Relu�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_2/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/dropout/Const�
dropout/dropout/MulMuldense_2/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout/dropout/Mulx
dropout/dropout/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform�
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dropout/dropout/GreaterEqual/y�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2
dropout/dropout/GreaterEqual�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout/dropout/Cast�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout/dropout/Mul_1�
IdentityIdentitydropout/dropout/Mul_1:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������1::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������1
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_4229O
Kwide_deep_embedding_2_embeddings_regularizer_square_readvariableop_resource
identity��Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpKwide_deep_embedding_2_embeddings_regularizer_square_readvariableop_resource*
_output_shapes
:	�*
dtype02D
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_2/embeddings/Regularizer/SquareSquareJwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�25
3wide_deep/embedding_2/embeddings/Regularizer/Square�
2wide_deep/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_2/embeddings/Regularizer/Const�
0wide_deep/embedding_2/embeddings/Regularizer/SumSum7wide_deep/embedding_2/embeddings/Regularizer/Square:y:0;wide_deep/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/Sum�
2wide_deep/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_2/embeddings/Regularizer/mul/x�
0wide_deep/embedding_2/embeddings/Regularizer/mulMul;wide_deep/embedding_2/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/mul�
IdentityIdentity4wide_deep/embedding_2/embeddings/Regularizer/mul:z:0C^wide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp
��
�
C__inference_wide_deep_layer_call_and_return_conditional_losses_3610

inputs
inputs_1
embedding_3515
embedding_1_3522
embedding_2_3529
embedding_3_3536
embedding_4_3543
linear_3550
linear_3552
dnn_3555
dnn_3557
dnn_3559
dnn_3561
dnn_3563
dnn_3565
dense_4_3568
dense_4_3570
identity��dense_4/StatefulPartitionedCall�dnn/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�#embedding_3/StatefulPartitionedCall�#embedding_4/StatefulPartitionedCall�linear/StatefulPartitionedCall�@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinputs_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice�
!embedding/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0embedding_3515*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_29052#
!embedding/StatefulPartitionedCall
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputs_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_1_3522*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_29362%
#embedding_1/StatefulPartitionedCall
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack�
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1�
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2�
strided_slice_2StridedSliceinputs_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_2_3529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_29672%
#embedding_2/StatefulPartitionedCall
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack�
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1�
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2�
strided_slice_3StridedSliceinputs_1strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3�
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_3_3536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_3_layer_call_and_return_conditional_losses_29982%
#embedding_3/StatefulPartitionedCall
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack�
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1�
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2�
strided_slice_4StridedSliceinputs_1strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4�
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_4:output:0embedding_4_3543*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_4_layer_call_and_return_conditional_losses_30292%
#embedding_4/StatefulPartitionedCalle
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/axis�
concatConcatV2*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0,embedding_4/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������(2
concati
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat_1/axis�
concat_1ConcatV2concat:output:0inputsconcat_1/axis:output:0*
N*
T0*'
_output_shapes
:���������12

concat_1�
linear/StatefulPartitionedCallStatefulPartitionedCallinputslinear_3550linear_3552*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_linear_layer_call_and_return_conditional_losses_30652 
linear/StatefulPartitionedCall�
dnn/StatefulPartitionedCallStatefulPartitionedCallconcat_1:output:0dnn_3555dnn_3557dnn_3559dnn_3561dnn_3563dnn_3565*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_dnn_layer_call_and_return_conditional_losses_31492
dnn/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall$dnn/StatefulPartitionedCall:output:0dense_4_3568dense_4_3570*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_32082!
dense_4/StatefulPartitionedCallS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x|
mulMulmul/x:output:0'linear/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/x�
mul_1Mulmul_1/x:output:0(dense_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
mul_1Y
addAddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_3515* 
_output_shapes
:
Ͷ*
dtype02B
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�
1wide_deep/embedding/embeddings/Regularizer/SquareSquareHwide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Ͷ23
1wide_deep/embedding/embeddings/Regularizer/Square�
0wide_deep/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       22
0wide_deep/embedding/embeddings/Regularizer/Const�
.wide_deep/embedding/embeddings/Regularizer/SumSum5wide_deep/embedding/embeddings/Regularizer/Square:y:09wide_deep/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/Sum�
0wide_deep/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��822
0wide_deep/embedding/embeddings/Regularizer/mul/x�
.wide_deep/embedding/embeddings/Regularizer/mulMul9wide_deep/embedding/embeddings/Regularizer/mul/x:output:07wide_deep/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/mul�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_3522* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_1/embeddings/Regularizer/SquareSquareJwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_1/embeddings/Regularizer/Square�
2wide_deep/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_1/embeddings/Regularizer/Const�
0wide_deep/embedding_1/embeddings/Regularizer/SumSum7wide_deep/embedding_1/embeddings/Regularizer/Square:y:0;wide_deep/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/Sum�
2wide_deep/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_1/embeddings/Regularizer/mul/x�
0wide_deep/embedding_1/embeddings/Regularizer/mulMul;wide_deep/embedding_1/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/mul�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_3529*
_output_shapes
:	�*
dtype02D
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_2/embeddings/Regularizer/SquareSquareJwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�25
3wide_deep/embedding_2/embeddings/Regularizer/Square�
2wide_deep/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_2/embeddings/Regularizer/Const�
0wide_deep/embedding_2/embeddings/Regularizer/SumSum7wide_deep/embedding_2/embeddings/Regularizer/Square:y:0;wide_deep/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/Sum�
2wide_deep/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_2/embeddings/Regularizer/mul/x�
0wide_deep/embedding_2/embeddings/Regularizer/mulMul;wide_deep/embedding_2/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/mul�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_3_3536*
_output_shapes

:*
dtype02D
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_3/embeddings/Regularizer/SquareSquareJwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:25
3wide_deep/embedding_3/embeddings/Regularizer/Square�
2wide_deep/embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_3/embeddings/Regularizer/Const�
0wide_deep/embedding_3/embeddings/Regularizer/SumSum7wide_deep/embedding_3/embeddings/Regularizer/Square:y:0;wide_deep/embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/Sum�
2wide_deep/embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_3/embeddings/Regularizer/mul/x�
0wide_deep/embedding_3/embeddings/Regularizer/mulMul;wide_deep/embedding_3/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/mul�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_4_3543* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_4/embeddings/Regularizer/SquareSquareJwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_4/embeddings/Regularizer/Square�
2wide_deep/embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_4/embeddings/Regularizer/Const�
0wide_deep/embedding_4/embeddings/Regularizer/SumSum7wide_deep/embedding_4/embeddings/Regularizer/Square:y:0;wide_deep/embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/Sum�
2wide_deep/embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_4/embeddings/Regularizer/mul/x�
0wide_deep/embedding_4/embeddings/Regularizer/mulMul;wide_deep/embedding_4/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/mul�
IdentityIdentitySigmoid:y:0 ^dense_4/StatefulPartitionedCall^dnn/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall^linear/StatefulPartitionedCallA^wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*u
_input_shapesd
b:���������	:���������:::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2:
dnn/StatefulPartitionedCalldnn/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2@
linear/StatefulPartitionedCalllinear/StatefulPartitionedCall2�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
p
*__inference_embedding_2_layer_call_fn_4335

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_29672
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
p
*__inference_embedding_4_layer_call_fn_4391

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_4_layer_call_and_return_conditional_losses_30292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_0_4207M
Iwide_deep_embedding_embeddings_regularizer_square_readvariableop_resource
identity��@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpIwide_deep_embedding_embeddings_regularizer_square_readvariableop_resource* 
_output_shapes
:
Ͷ*
dtype02B
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�
1wide_deep/embedding/embeddings/Regularizer/SquareSquareHwide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Ͷ23
1wide_deep/embedding/embeddings/Regularizer/Square�
0wide_deep/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       22
0wide_deep/embedding/embeddings/Regularizer/Const�
.wide_deep/embedding/embeddings/Regularizer/SumSum5wide_deep/embedding/embeddings/Regularizer/Square:y:09wide_deep/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/Sum�
0wide_deep/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��822
0wide_deep/embedding/embeddings/Regularizer/mul/x�
.wide_deep/embedding/embeddings/Regularizer/mulMul9wide_deep/embedding/embeddings/Regularizer/mul/x:output:07wide_deep/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/mul�
IdentityIdentity2wide_deep/embedding/embeddings/Regularizer/mul:z:0A^wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp
�&
�
=__inference_dnn_layer_call_and_return_conditional_losses_3123

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	1�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_1/Relu�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_2/Relus
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dropout/dropout/Const�
dropout/dropout/MulMuldense_2/Relu:activations:0dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dropout/dropout/Mulx
dropout/dropout/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
dropout/dropout/Shape�
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype02.
,dropout/dropout/random_uniform/RandomUniform�
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
dropout/dropout/GreaterEqual/y�
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2
dropout/dropout/GreaterEqual�
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dropout/dropout/Cast�
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dropout/dropout/Mul_1�
IdentityIdentitydropout/dropout/Mul_1:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������1::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������1
 
_user_specified_nameinputs
�
�
=__inference_dnn_layer_call_and_return_conditional_losses_4105

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	1�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_1/Relu�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_2/Relu~
dropout/IdentityIdentitydense_2/Relu:activations:0*
T0*'
_output_shapes
:���������@2
dropout/Identity�
IdentityIdentitydropout/Identity:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������1::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������1
 
_user_specified_nameinputs
�

�
@__inference_linear_layer_call_and_return_conditional_losses_3065

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity��dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/BiasAdd�
IdentityIdentitydense_3/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������	::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
"__inference_dnn_layer_call_fn_4122

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_dnn_layer_call_and_return_conditional_losses_31232
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������1::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������1
 
_user_specified_nameinputs
��
�
C__inference_wide_deep_layer_call_and_return_conditional_losses_3974
inputs_0
inputs_16
2embedding_embedding_lookup_readvariableop_resource8
4embedding_1_embedding_lookup_readvariableop_resource8
4embedding_2_embedding_lookup_readvariableop_resource8
4embedding_3_embedding_lookup_readvariableop_resource8
4embedding_4_embedding_lookup_readvariableop_resource1
-linear_dense_3_matmul_readvariableop_resource2
.linear_dense_3_biasadd_readvariableop_resource,
(dnn_dense_matmul_readvariableop_resource-
)dnn_dense_biasadd_readvariableop_resource.
*dnn_dense_1_matmul_readvariableop_resource/
+dnn_dense_1_biasadd_readvariableop_resource.
*dnn_dense_2_matmul_readvariableop_resource/
+dnn_dense_2_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity��dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp� dnn/dense/BiasAdd/ReadVariableOp�dnn/dense/MatMul/ReadVariableOp�"dnn/dense_1/BiasAdd/ReadVariableOp�!dnn/dense_1/MatMul/ReadVariableOp�"dnn/dense_2/BiasAdd/ReadVariableOp�!dnn/dense_2/MatMul/ReadVariableOp�)embedding/embedding_lookup/ReadVariableOp�+embedding_1/embedding_lookup/ReadVariableOp�+embedding_2/embedding_lookup/ReadVariableOp�+embedding_3/embedding_lookup/ReadVariableOp�+embedding_4/embedding_lookup/ReadVariableOp�%linear/dense_3/BiasAdd/ReadVariableOp�$linear/dense_3/MatMul/ReadVariableOp�@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinputs_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice�
)embedding/embedding_lookup/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
Ͷ*
dtype02+
)embedding/embedding_lookup/ReadVariableOp�
embedding/embedding_lookup/axisConst*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2!
embedding/embedding_lookup/axis�
embedding/embedding_lookupGatherV21embedding/embedding_lookup/ReadVariableOp:value:0strided_slice:output:0(embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding/embedding_lookup�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*'
_output_shapes
:���������2%
#embedding/embedding_lookup/Identity
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputs_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1�
+embedding_1/embedding_lookup/ReadVariableOpReadVariableOp4embedding_1_embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+embedding_1/embedding_lookup/ReadVariableOp�
!embedding_1/embedding_lookup/axisConst*>
_class4
20loc:@embedding_1/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2#
!embedding_1/embedding_lookup/axis�
embedding_1/embedding_lookupGatherV23embedding_1/embedding_lookup/ReadVariableOp:value:0strided_slice_1:output:0*embedding_1/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*>
_class4
20loc:@embedding_1/embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_1/embedding_lookup�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*'
_output_shapes
:���������2'
%embedding_1/embedding_lookup/Identity
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack�
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1�
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2�
strided_slice_2StridedSliceinputs_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2�
+embedding_2/embedding_lookup/ReadVariableOpReadVariableOp4embedding_2_embedding_lookup_readvariableop_resource*
_output_shapes
:	�*
dtype02-
+embedding_2/embedding_lookup/ReadVariableOp�
!embedding_2/embedding_lookup/axisConst*>
_class4
20loc:@embedding_2/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2#
!embedding_2/embedding_lookup/axis�
embedding_2/embedding_lookupGatherV23embedding_2/embedding_lookup/ReadVariableOp:value:0strided_slice_2:output:0*embedding_2/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*>
_class4
20loc:@embedding_2/embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_2/embedding_lookup�
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*'
_output_shapes
:���������2'
%embedding_2/embedding_lookup/Identity
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack�
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1�
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2�
strided_slice_3StridedSliceinputs_1strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3�
+embedding_3/embedding_lookup/ReadVariableOpReadVariableOp4embedding_3_embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype02-
+embedding_3/embedding_lookup/ReadVariableOp�
!embedding_3/embedding_lookup/axisConst*>
_class4
20loc:@embedding_3/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2#
!embedding_3/embedding_lookup/axis�
embedding_3/embedding_lookupGatherV23embedding_3/embedding_lookup/ReadVariableOp:value:0strided_slice_3:output:0*embedding_3/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*>
_class4
20loc:@embedding_3/embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_3/embedding_lookup�
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0*
T0*'
_output_shapes
:���������2'
%embedding_3/embedding_lookup/Identity
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack�
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1�
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2�
strided_slice_4StridedSliceinputs_1strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4�
+embedding_4/embedding_lookup/ReadVariableOpReadVariableOp4embedding_4_embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+embedding_4/embedding_lookup/ReadVariableOp�
!embedding_4/embedding_lookup/axisConst*>
_class4
20loc:@embedding_4/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2#
!embedding_4/embedding_lookup/axis�
embedding_4/embedding_lookupGatherV23embedding_4/embedding_lookup/ReadVariableOp:value:0strided_slice_4:output:0*embedding_4/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*>
_class4
20loc:@embedding_4/embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_4/embedding_lookup�
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0*
T0*'
_output_shapes
:���������2'
%embedding_4/embedding_lookup/Identitye
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/axis�
concatConcatV2,embedding/embedding_lookup/Identity:output:0.embedding_1/embedding_lookup/Identity:output:0.embedding_2/embedding_lookup/Identity:output:0.embedding_3/embedding_lookup/Identity:output:0.embedding_4/embedding_lookup/Identity:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������(2
concati
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat_1/axis�
concat_1ConcatV2concat:output:0inputs_0concat_1/axis:output:0*
N*
T0*'
_output_shapes
:���������12

concat_1�
$linear/dense_3/MatMul/ReadVariableOpReadVariableOp-linear_dense_3_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02&
$linear/dense_3/MatMul/ReadVariableOp�
linear/dense_3/MatMulMatMulinputs_0,linear/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
linear/dense_3/MatMul�
%linear/dense_3/BiasAdd/ReadVariableOpReadVariableOp.linear_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%linear/dense_3/BiasAdd/ReadVariableOp�
linear/dense_3/BiasAddBiasAddlinear/dense_3/MatMul:product:0-linear/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
linear/dense_3/BiasAdd�
dnn/dense/MatMul/ReadVariableOpReadVariableOp(dnn_dense_matmul_readvariableop_resource*
_output_shapes
:	1�*
dtype02!
dnn/dense/MatMul/ReadVariableOp�
dnn/dense/MatMulMatMulconcat_1:output:0'dnn/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dnn/dense/MatMul�
 dnn/dense/BiasAdd/ReadVariableOpReadVariableOp)dnn_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 dnn/dense/BiasAdd/ReadVariableOp�
dnn/dense/BiasAddBiasAdddnn/dense/MatMul:product:0(dnn/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dnn/dense/BiasAddw
dnn/dense/ReluReludnn/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dnn/dense/Relu�
!dnn/dense_1/MatMul/ReadVariableOpReadVariableOp*dnn_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!dnn/dense_1/MatMul/ReadVariableOp�
dnn/dense_1/MatMulMatMuldnn/dense/Relu:activations:0)dnn/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dnn/dense_1/MatMul�
"dnn/dense_1/BiasAdd/ReadVariableOpReadVariableOp+dnn_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"dnn/dense_1/BiasAdd/ReadVariableOp�
dnn/dense_1/BiasAddBiasAdddnn/dense_1/MatMul:product:0*dnn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dnn/dense_1/BiasAdd}
dnn/dense_1/ReluReludnn/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dnn/dense_1/Relu�
!dnn/dense_2/MatMul/ReadVariableOpReadVariableOp*dnn_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02#
!dnn/dense_2/MatMul/ReadVariableOp�
dnn/dense_2/MatMulMatMuldnn/dense_1/Relu:activations:0)dnn/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dnn/dense_2/MatMul�
"dnn/dense_2/BiasAdd/ReadVariableOpReadVariableOp+dnn_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"dnn/dense_2/BiasAdd/ReadVariableOp�
dnn/dense_2/BiasAddBiasAdddnn/dense_2/MatMul:product:0*dnn/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dnn/dense_2/BiasAdd|
dnn/dense_2/ReluReludnn/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dnn/dense_2/Relu�
dnn/dropout/IdentityIdentitydnn/dense_2/Relu:activations:0*
T0*'
_output_shapes
:���������@2
dnn/dropout/Identity�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMuldnn/dropout/Identity:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_4/BiasAddS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xt
mulMulmul/x:output:0linear/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/xs
mul_1Mulmul_1/x:output:0dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
mul_1Y
addAddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
Ͷ*
dtype02B
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�
1wide_deep/embedding/embeddings/Regularizer/SquareSquareHwide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Ͷ23
1wide_deep/embedding/embeddings/Regularizer/Square�
0wide_deep/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       22
0wide_deep/embedding/embeddings/Regularizer/Const�
.wide_deep/embedding/embeddings/Regularizer/SumSum5wide_deep/embedding/embeddings/Regularizer/Square:y:09wide_deep/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/Sum�
0wide_deep/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��822
0wide_deep/embedding/embeddings/Regularizer/mul/x�
.wide_deep/embedding/embeddings/Regularizer/mulMul9wide_deep/embedding/embeddings/Regularizer/mul/x:output:07wide_deep/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/mul�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp4embedding_1_embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_1/embeddings/Regularizer/SquareSquareJwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_1/embeddings/Regularizer/Square�
2wide_deep/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_1/embeddings/Regularizer/Const�
0wide_deep/embedding_1/embeddings/Regularizer/SumSum7wide_deep/embedding_1/embeddings/Regularizer/Square:y:0;wide_deep/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/Sum�
2wide_deep/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_1/embeddings/Regularizer/mul/x�
0wide_deep/embedding_1/embeddings/Regularizer/mulMul;wide_deep/embedding_1/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/mul�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp4embedding_2_embedding_lookup_readvariableop_resource*
_output_shapes
:	�*
dtype02D
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_2/embeddings/Regularizer/SquareSquareJwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�25
3wide_deep/embedding_2/embeddings/Regularizer/Square�
2wide_deep/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_2/embeddings/Regularizer/Const�
0wide_deep/embedding_2/embeddings/Regularizer/SumSum7wide_deep/embedding_2/embeddings/Regularizer/Square:y:0;wide_deep/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/Sum�
2wide_deep/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_2/embeddings/Regularizer/mul/x�
0wide_deep/embedding_2/embeddings/Regularizer/mulMul;wide_deep/embedding_2/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/mul�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp4embedding_3_embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype02D
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_3/embeddings/Regularizer/SquareSquareJwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:25
3wide_deep/embedding_3/embeddings/Regularizer/Square�
2wide_deep/embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_3/embeddings/Regularizer/Const�
0wide_deep/embedding_3/embeddings/Regularizer/SumSum7wide_deep/embedding_3/embeddings/Regularizer/Square:y:0;wide_deep/embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/Sum�
2wide_deep/embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_3/embeddings/Regularizer/mul/x�
0wide_deep/embedding_3/embeddings/Regularizer/mulMul;wide_deep/embedding_3/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/mul�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp4embedding_4_embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_4/embeddings/Regularizer/SquareSquareJwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_4/embeddings/Regularizer/Square�
2wide_deep/embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_4/embeddings/Regularizer/Const�
0wide_deep/embedding_4/embeddings/Regularizer/SumSum7wide_deep/embedding_4/embeddings/Regularizer/Square:y:0;wide_deep/embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/Sum�
2wide_deep/embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_4/embeddings/Regularizer/mul/x�
0wide_deep/embedding_4/embeddings/Regularizer/mulMul;wide_deep/embedding_4/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/mul�
IdentityIdentitySigmoid:y:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp!^dnn/dense/BiasAdd/ReadVariableOp ^dnn/dense/MatMul/ReadVariableOp#^dnn/dense_1/BiasAdd/ReadVariableOp"^dnn/dense_1/MatMul/ReadVariableOp#^dnn/dense_2/BiasAdd/ReadVariableOp"^dnn/dense_2/MatMul/ReadVariableOp*^embedding/embedding_lookup/ReadVariableOp,^embedding_1/embedding_lookup/ReadVariableOp,^embedding_2/embedding_lookup/ReadVariableOp,^embedding_3/embedding_lookup/ReadVariableOp,^embedding_4/embedding_lookup/ReadVariableOp&^linear/dense_3/BiasAdd/ReadVariableOp%^linear/dense_3/MatMul/ReadVariableOpA^wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*u
_input_shapesd
b:���������	:���������:::::::::::::::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2D
 dnn/dense/BiasAdd/ReadVariableOp dnn/dense/BiasAdd/ReadVariableOp2B
dnn/dense/MatMul/ReadVariableOpdnn/dense/MatMul/ReadVariableOp2H
"dnn/dense_1/BiasAdd/ReadVariableOp"dnn/dense_1/BiasAdd/ReadVariableOp2F
!dnn/dense_1/MatMul/ReadVariableOp!dnn/dense_1/MatMul/ReadVariableOp2H
"dnn/dense_2/BiasAdd/ReadVariableOp"dnn/dense_2/BiasAdd/ReadVariableOp2F
!dnn/dense_2/MatMul/ReadVariableOp!dnn/dense_2/MatMul/ReadVariableOp2V
)embedding/embedding_lookup/ReadVariableOp)embedding/embedding_lookup/ReadVariableOp2Z
+embedding_1/embedding_lookup/ReadVariableOp+embedding_1/embedding_lookup/ReadVariableOp2Z
+embedding_2/embedding_lookup/ReadVariableOp+embedding_2/embedding_lookup/ReadVariableOp2Z
+embedding_3/embedding_lookup/ReadVariableOp+embedding_3/embedding_lookup/ReadVariableOp2Z
+embedding_4/embedding_lookup/ReadVariableOp+embedding_4/embedding_lookup/ReadVariableOp2N
%linear/dense_3/BiasAdd/ReadVariableOp%linear/dense_3/BiasAdd/ReadVariableOp2L
$linear/dense_3/MatMul/ReadVariableOp$linear/dense_3/MatMul/ReadVariableOp2�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:���������	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
(__inference_wide_deep_layer_call_fn_4010
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_wide_deep_layer_call_and_return_conditional_losses_34712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*u
_input_shapesd
b:���������	:���������:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�u
�
__inference__traced_save_4583
file_prefix7
3savev2_wide_deep_dense_4_kernel_read_readvariableop5
1savev2_wide_deep_dense_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop=
9savev2_wide_deep_embedding_embeddings_read_readvariableop?
;savev2_wide_deep_embedding_1_embeddings_read_readvariableop?
;savev2_wide_deep_embedding_2_embeddings_read_readvariableop?
;savev2_wide_deep_embedding_3_embeddings_read_readvariableop?
;savev2_wide_deep_embedding_4_embeddings_read_readvariableop9
5savev2_wide_deep_dnn_dense_kernel_read_readvariableop7
3savev2_wide_deep_dnn_dense_bias_read_readvariableop;
7savev2_wide_deep_dnn_dense_1_kernel_read_readvariableop9
5savev2_wide_deep_dnn_dense_1_bias_read_readvariableop;
7savev2_wide_deep_dnn_dense_2_kernel_read_readvariableop9
5savev2_wide_deep_dnn_dense_2_bias_read_readvariableop>
:savev2_wide_deep_linear_dense_3_kernel_read_readvariableop<
8savev2_wide_deep_linear_dense_3_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop-
)savev2_true_negatives_read_readvariableop.
*savev2_false_positives_read_readvariableop.
*savev2_false_negatives_read_readvariableop>
:savev2_adam_wide_deep_dense_4_kernel_m_read_readvariableop<
8savev2_adam_wide_deep_dense_4_bias_m_read_readvariableopD
@savev2_adam_wide_deep_embedding_embeddings_m_read_readvariableopF
Bsavev2_adam_wide_deep_embedding_1_embeddings_m_read_readvariableopF
Bsavev2_adam_wide_deep_embedding_2_embeddings_m_read_readvariableopF
Bsavev2_adam_wide_deep_embedding_3_embeddings_m_read_readvariableopF
Bsavev2_adam_wide_deep_embedding_4_embeddings_m_read_readvariableop@
<savev2_adam_wide_deep_dnn_dense_kernel_m_read_readvariableop>
:savev2_adam_wide_deep_dnn_dense_bias_m_read_readvariableopB
>savev2_adam_wide_deep_dnn_dense_1_kernel_m_read_readvariableop@
<savev2_adam_wide_deep_dnn_dense_1_bias_m_read_readvariableopB
>savev2_adam_wide_deep_dnn_dense_2_kernel_m_read_readvariableop@
<savev2_adam_wide_deep_dnn_dense_2_bias_m_read_readvariableopE
Asavev2_adam_wide_deep_linear_dense_3_kernel_m_read_readvariableopC
?savev2_adam_wide_deep_linear_dense_3_bias_m_read_readvariableop>
:savev2_adam_wide_deep_dense_4_kernel_v_read_readvariableop<
8savev2_adam_wide_deep_dense_4_bias_v_read_readvariableopD
@savev2_adam_wide_deep_embedding_embeddings_v_read_readvariableopF
Bsavev2_adam_wide_deep_embedding_1_embeddings_v_read_readvariableopF
Bsavev2_adam_wide_deep_embedding_2_embeddings_v_read_readvariableopF
Bsavev2_adam_wide_deep_embedding_3_embeddings_v_read_readvariableopF
Bsavev2_adam_wide_deep_embedding_4_embeddings_v_read_readvariableop@
<savev2_adam_wide_deep_dnn_dense_kernel_v_read_readvariableop>
:savev2_adam_wide_deep_dnn_dense_bias_v_read_readvariableopB
>savev2_adam_wide_deep_dnn_dense_1_kernel_v_read_readvariableop@
<savev2_adam_wide_deep_dnn_dense_1_bias_v_read_readvariableopB
>savev2_adam_wide_deep_dnn_dense_2_kernel_v_read_readvariableop@
<savev2_adam_wide_deep_dnn_dense_2_bias_v_read_readvariableopE
Asavev2_adam_wide_deep_linear_dense_3_kernel_v_read_readvariableopC
?savev2_adam_wide_deep_linear_dense_3_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*�
value�B�9B-final_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB+final_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBIfinal_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGfinal_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIfinal_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGfinal_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*�
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_wide_deep_dense_4_kernel_read_readvariableop1savev2_wide_deep_dense_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop9savev2_wide_deep_embedding_embeddings_read_readvariableop;savev2_wide_deep_embedding_1_embeddings_read_readvariableop;savev2_wide_deep_embedding_2_embeddings_read_readvariableop;savev2_wide_deep_embedding_3_embeddings_read_readvariableop;savev2_wide_deep_embedding_4_embeddings_read_readvariableop5savev2_wide_deep_dnn_dense_kernel_read_readvariableop3savev2_wide_deep_dnn_dense_bias_read_readvariableop7savev2_wide_deep_dnn_dense_1_kernel_read_readvariableop5savev2_wide_deep_dnn_dense_1_bias_read_readvariableop7savev2_wide_deep_dnn_dense_2_kernel_read_readvariableop5savev2_wide_deep_dnn_dense_2_bias_read_readvariableop:savev2_wide_deep_linear_dense_3_kernel_read_readvariableop8savev2_wide_deep_linear_dense_3_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop)savev2_true_negatives_read_readvariableop*savev2_false_positives_read_readvariableop*savev2_false_negatives_read_readvariableop:savev2_adam_wide_deep_dense_4_kernel_m_read_readvariableop8savev2_adam_wide_deep_dense_4_bias_m_read_readvariableop@savev2_adam_wide_deep_embedding_embeddings_m_read_readvariableopBsavev2_adam_wide_deep_embedding_1_embeddings_m_read_readvariableopBsavev2_adam_wide_deep_embedding_2_embeddings_m_read_readvariableopBsavev2_adam_wide_deep_embedding_3_embeddings_m_read_readvariableopBsavev2_adam_wide_deep_embedding_4_embeddings_m_read_readvariableop<savev2_adam_wide_deep_dnn_dense_kernel_m_read_readvariableop:savev2_adam_wide_deep_dnn_dense_bias_m_read_readvariableop>savev2_adam_wide_deep_dnn_dense_1_kernel_m_read_readvariableop<savev2_adam_wide_deep_dnn_dense_1_bias_m_read_readvariableop>savev2_adam_wide_deep_dnn_dense_2_kernel_m_read_readvariableop<savev2_adam_wide_deep_dnn_dense_2_bias_m_read_readvariableopAsavev2_adam_wide_deep_linear_dense_3_kernel_m_read_readvariableop?savev2_adam_wide_deep_linear_dense_3_bias_m_read_readvariableop:savev2_adam_wide_deep_dense_4_kernel_v_read_readvariableop8savev2_adam_wide_deep_dense_4_bias_v_read_readvariableop@savev2_adam_wide_deep_embedding_embeddings_v_read_readvariableopBsavev2_adam_wide_deep_embedding_1_embeddings_v_read_readvariableopBsavev2_adam_wide_deep_embedding_2_embeddings_v_read_readvariableopBsavev2_adam_wide_deep_embedding_3_embeddings_v_read_readvariableopBsavev2_adam_wide_deep_embedding_4_embeddings_v_read_readvariableop<savev2_adam_wide_deep_dnn_dense_kernel_v_read_readvariableop:savev2_adam_wide_deep_dnn_dense_bias_v_read_readvariableop>savev2_adam_wide_deep_dnn_dense_1_kernel_v_read_readvariableop<savev2_adam_wide_deep_dnn_dense_1_bias_v_read_readvariableop>savev2_adam_wide_deep_dnn_dense_2_kernel_v_read_readvariableop<savev2_adam_wide_deep_dnn_dense_2_bias_v_read_readvariableopAsavev2_adam_wide_deep_linear_dense_3_kernel_v_read_readvariableop?savev2_adam_wide_deep_linear_dense_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *G
dtypes=
;29	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :@:: : : : : :
Ͷ:
��:	�::
��:	1�:�:
��:�:	�@:@:	:: : :�:�:�:�:@::
Ͷ:
��:	�::
��:	1�:�:
��:�:	�@:@:	::@::
Ͷ:
��:	�::
��:	1�:�:
��:�:	�@:@:	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
Ͷ:&	"
 
_output_shapes
:
��:%
!

_output_shapes
:	�:$ 

_output_shapes

::&"
 
_output_shapes
:
��:%!

_output_shapes
:	1�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:$ 

_output_shapes

:@: 

_output_shapes
::&"
 
_output_shapes
:
Ͷ:&"
 
_output_shapes
:
��:%!

_output_shapes
:	�:$  

_output_shapes

::&!"
 
_output_shapes
:
��:%"!

_output_shapes
:	1�:!#

_output_shapes	
:�:&$"
 
_output_shapes
:
��:!%

_output_shapes	
:�:%&!

_output_shapes
:	�@: '

_output_shapes
:@:$( 

_output_shapes

:	: )

_output_shapes
::$* 

_output_shapes

:@: +

_output_shapes
::&,"
 
_output_shapes
:
Ͷ:&-"
 
_output_shapes
:
��:%.!

_output_shapes
:	�:$/ 

_output_shapes

::&0"
 
_output_shapes
:
��:%1!

_output_shapes
:	1�:!2

_output_shapes	
:�:&3"
 
_output_shapes
:
��:!4

_output_shapes	
:�:%5!

_output_shapes
:	�@: 6

_output_shapes
:@:$7 

_output_shapes

:	: 8

_output_shapes
::9

_output_shapes
: 
�
�
(__inference_wide_deep_layer_call_fn_3643
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_wide_deep_layer_call_and_return_conditional_losses_36102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*u
_input_shapesd
b:���������	:���������:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
�
E__inference_embedding_3_layer_call_and_return_conditional_losses_2998

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype02!
embedding_lookup/ReadVariableOp�
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis�
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:���������2
embedding_lookup/Identity�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype02D
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_3/embeddings/Regularizer/SquareSquareJwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:25
3wide_deep/embedding_3/embeddings/Regularizer/Square�
2wide_deep/embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_3/embeddings/Regularizer/Const�
0wide_deep/embedding_3/embeddings/Regularizer/SumSum7wide_deep/embedding_3/embeddings/Regularizer/Square:y:0;wide_deep/embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/Sum�
2wide_deep/embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_3/embeddings/Regularizer/mul/x�
0wide_deep/embedding_3/embeddings/Regularizer/mulMul;wide_deep/embedding_3/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/mul�
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOpC^wide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp2�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
C__inference_wide_deep_layer_call_and_return_conditional_losses_3261
input_1
input_2
embedding_2914
embedding_1_2945
embedding_2_2976
embedding_3_3007
embedding_4_3038
linear_3085
linear_3087
dnn_3185
dnn_3187
dnn_3189
dnn_3191
dnn_3193
dnn_3195
dense_4_3219
dense_4_3221
identity��dense_4/StatefulPartitionedCall�dnn/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�#embedding_3/StatefulPartitionedCall�#embedding_4/StatefulPartitionedCall�linear/StatefulPartitionedCall�@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinput_2strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice�
!embedding/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0embedding_2914*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_29052#
!embedding/StatefulPartitionedCall
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinput_2strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_1_2945*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_29362%
#embedding_1/StatefulPartitionedCall
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack�
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1�
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2�
strided_slice_2StridedSliceinput_2strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_2_2976*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_29672%
#embedding_2/StatefulPartitionedCall
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack�
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1�
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2�
strided_slice_3StridedSliceinput_2strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3�
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_3_3007*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_3_layer_call_and_return_conditional_losses_29982%
#embedding_3/StatefulPartitionedCall
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack�
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1�
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2�
strided_slice_4StridedSliceinput_2strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4�
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_4:output:0embedding_4_3038*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_4_layer_call_and_return_conditional_losses_30292%
#embedding_4/StatefulPartitionedCalle
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/axis�
concatConcatV2*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0,embedding_4/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������(2
concati
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat_1/axis�
concat_1ConcatV2concat:output:0input_1concat_1/axis:output:0*
N*
T0*'
_output_shapes
:���������12

concat_1�
linear/StatefulPartitionedCallStatefulPartitionedCallinput_1linear_3085linear_3087*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_linear_layer_call_and_return_conditional_losses_30552 
linear/StatefulPartitionedCall�
dnn/StatefulPartitionedCallStatefulPartitionedCallconcat_1:output:0dnn_3185dnn_3187dnn_3189dnn_3191dnn_3193dnn_3195*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_dnn_layer_call_and_return_conditional_losses_31232
dnn/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall$dnn/StatefulPartitionedCall:output:0dense_4_3219dense_4_3221*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_32082!
dense_4/StatefulPartitionedCallS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x|
mulMulmul/x:output:0'linear/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/x�
mul_1Mulmul_1/x:output:0(dense_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
mul_1Y
addAddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2914* 
_output_shapes
:
Ͷ*
dtype02B
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�
1wide_deep/embedding/embeddings/Regularizer/SquareSquareHwide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Ͷ23
1wide_deep/embedding/embeddings/Regularizer/Square�
0wide_deep/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       22
0wide_deep/embedding/embeddings/Regularizer/Const�
.wide_deep/embedding/embeddings/Regularizer/SumSum5wide_deep/embedding/embeddings/Regularizer/Square:y:09wide_deep/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/Sum�
0wide_deep/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��822
0wide_deep/embedding/embeddings/Regularizer/mul/x�
.wide_deep/embedding/embeddings/Regularizer/mulMul9wide_deep/embedding/embeddings/Regularizer/mul/x:output:07wide_deep/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/mul�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_2945* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_1/embeddings/Regularizer/SquareSquareJwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_1/embeddings/Regularizer/Square�
2wide_deep/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_1/embeddings/Regularizer/Const�
0wide_deep/embedding_1/embeddings/Regularizer/SumSum7wide_deep/embedding_1/embeddings/Regularizer/Square:y:0;wide_deep/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/Sum�
2wide_deep/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_1/embeddings/Regularizer/mul/x�
0wide_deep/embedding_1/embeddings/Regularizer/mulMul;wide_deep/embedding_1/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/mul�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_2976*
_output_shapes
:	�*
dtype02D
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_2/embeddings/Regularizer/SquareSquareJwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�25
3wide_deep/embedding_2/embeddings/Regularizer/Square�
2wide_deep/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_2/embeddings/Regularizer/Const�
0wide_deep/embedding_2/embeddings/Regularizer/SumSum7wide_deep/embedding_2/embeddings/Regularizer/Square:y:0;wide_deep/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/Sum�
2wide_deep/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_2/embeddings/Regularizer/mul/x�
0wide_deep/embedding_2/embeddings/Regularizer/mulMul;wide_deep/embedding_2/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/mul�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_3_3007*
_output_shapes

:*
dtype02D
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_3/embeddings/Regularizer/SquareSquareJwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:25
3wide_deep/embedding_3/embeddings/Regularizer/Square�
2wide_deep/embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_3/embeddings/Regularizer/Const�
0wide_deep/embedding_3/embeddings/Regularizer/SumSum7wide_deep/embedding_3/embeddings/Regularizer/Square:y:0;wide_deep/embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/Sum�
2wide_deep/embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_3/embeddings/Regularizer/mul/x�
0wide_deep/embedding_3/embeddings/Regularizer/mulMul;wide_deep/embedding_3/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/mul�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_4_3038* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_4/embeddings/Regularizer/SquareSquareJwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_4/embeddings/Regularizer/Square�
2wide_deep/embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_4/embeddings/Regularizer/Const�
0wide_deep/embedding_4/embeddings/Regularizer/SumSum7wide_deep/embedding_4/embeddings/Regularizer/Square:y:0;wide_deep/embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/Sum�
2wide_deep/embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_4/embeddings/Regularizer/mul/x�
0wide_deep/embedding_4/embeddings/Regularizer/mulMul;wide_deep/embedding_4/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/mul�
IdentityIdentitySigmoid:y:0 ^dense_4/StatefulPartitionedCall^dnn/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall^linear/StatefulPartitionedCallA^wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*u
_input_shapesd
b:���������	:���������:::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2:
dnn/StatefulPartitionedCalldnn/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2@
linear/StatefulPartitionedCalllinear/StatefulPartitionedCall2�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
z
%__inference_linear_layer_call_fn_4177

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_linear_layer_call_and_return_conditional_losses_30652
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������	::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
"__inference_dnn_layer_call_fn_4139

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_dnn_layer_call_and_return_conditional_losses_31492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������1::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������1
 
_user_specified_nameinputs
�
p
*__inference_embedding_3_layer_call_fn_4363

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_3_layer_call_and_return_conditional_losses_29982
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_embedding_4_layer_call_and_return_conditional_losses_4384

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02!
embedding_lookup/ReadVariableOp�
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis�
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:���������2
embedding_lookup/Identity�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_4/embeddings/Regularizer/SquareSquareJwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_4/embeddings/Regularizer/Square�
2wide_deep/embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_4/embeddings/Regularizer/Const�
0wide_deep/embedding_4/embeddings/Regularizer/SumSum7wide_deep/embedding_4/embeddings/Regularizer/Square:y:0;wide_deep/embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/Sum�
2wide_deep/embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_4/embeddings/Regularizer/mul/x�
0wide_deep/embedding_4/embeddings/Regularizer/mulMul;wide_deep/embedding_4/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/mul�
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOpC^wide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp2�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
A__inference_dense_4_layer_call_and_return_conditional_losses_4187

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_4218O
Kwide_deep_embedding_1_embeddings_regularizer_square_readvariableop_resource
identity��Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpKwide_deep_embedding_1_embeddings_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_1/embeddings/Regularizer/SquareSquareJwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_1/embeddings/Regularizer/Square�
2wide_deep/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_1/embeddings/Regularizer/Const�
0wide_deep/embedding_1/embeddings/Regularizer/SumSum7wide_deep/embedding_1/embeddings/Regularizer/Square:y:0;wide_deep/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/Sum�
2wide_deep/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_1/embeddings/Regularizer/mul/x�
0wide_deep/embedding_1/embeddings/Regularizer/mulMul;wide_deep/embedding_1/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/mul�
IdentityIdentity4wide_deep/embedding_1/embeddings/Regularizer/mul:z:0C^wide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp
��
�!
 __inference__traced_restore_4761
file_prefix-
)assignvariableop_wide_deep_dense_4_kernel-
)assignvariableop_1_wide_deep_dense_4_bias 
assignvariableop_2_adam_iter"
assignvariableop_3_adam_beta_1"
assignvariableop_4_adam_beta_2!
assignvariableop_5_adam_decay)
%assignvariableop_6_adam_learning_rate5
1assignvariableop_7_wide_deep_embedding_embeddings7
3assignvariableop_8_wide_deep_embedding_1_embeddings7
3assignvariableop_9_wide_deep_embedding_2_embeddings8
4assignvariableop_10_wide_deep_embedding_3_embeddings8
4assignvariableop_11_wide_deep_embedding_4_embeddings2
.assignvariableop_12_wide_deep_dnn_dense_kernel0
,assignvariableop_13_wide_deep_dnn_dense_bias4
0assignvariableop_14_wide_deep_dnn_dense_1_kernel2
.assignvariableop_15_wide_deep_dnn_dense_1_bias4
0assignvariableop_16_wide_deep_dnn_dense_2_kernel2
.assignvariableop_17_wide_deep_dnn_dense_2_bias7
3assignvariableop_18_wide_deep_linear_dense_3_kernel5
1assignvariableop_19_wide_deep_linear_dense_3_bias
assignvariableop_20_total
assignvariableop_21_count&
"assignvariableop_22_true_positives&
"assignvariableop_23_true_negatives'
#assignvariableop_24_false_positives'
#assignvariableop_25_false_negatives7
3assignvariableop_26_adam_wide_deep_dense_4_kernel_m5
1assignvariableop_27_adam_wide_deep_dense_4_bias_m=
9assignvariableop_28_adam_wide_deep_embedding_embeddings_m?
;assignvariableop_29_adam_wide_deep_embedding_1_embeddings_m?
;assignvariableop_30_adam_wide_deep_embedding_2_embeddings_m?
;assignvariableop_31_adam_wide_deep_embedding_3_embeddings_m?
;assignvariableop_32_adam_wide_deep_embedding_4_embeddings_m9
5assignvariableop_33_adam_wide_deep_dnn_dense_kernel_m7
3assignvariableop_34_adam_wide_deep_dnn_dense_bias_m;
7assignvariableop_35_adam_wide_deep_dnn_dense_1_kernel_m9
5assignvariableop_36_adam_wide_deep_dnn_dense_1_bias_m;
7assignvariableop_37_adam_wide_deep_dnn_dense_2_kernel_m9
5assignvariableop_38_adam_wide_deep_dnn_dense_2_bias_m>
:assignvariableop_39_adam_wide_deep_linear_dense_3_kernel_m<
8assignvariableop_40_adam_wide_deep_linear_dense_3_bias_m7
3assignvariableop_41_adam_wide_deep_dense_4_kernel_v5
1assignvariableop_42_adam_wide_deep_dense_4_bias_v=
9assignvariableop_43_adam_wide_deep_embedding_embeddings_v?
;assignvariableop_44_adam_wide_deep_embedding_1_embeddings_v?
;assignvariableop_45_adam_wide_deep_embedding_2_embeddings_v?
;assignvariableop_46_adam_wide_deep_embedding_3_embeddings_v?
;assignvariableop_47_adam_wide_deep_embedding_4_embeddings_v9
5assignvariableop_48_adam_wide_deep_dnn_dense_kernel_v7
3assignvariableop_49_adam_wide_deep_dnn_dense_bias_v;
7assignvariableop_50_adam_wide_deep_dnn_dense_1_kernel_v9
5assignvariableop_51_adam_wide_deep_dnn_dense_1_bias_v;
7assignvariableop_52_adam_wide_deep_dnn_dense_2_kernel_v9
5assignvariableop_53_adam_wide_deep_dnn_dense_2_bias_v>
:assignvariableop_54_adam_wide_deep_linear_dense_3_kernel_v<
8assignvariableop_55_adam_wide_deep_linear_dense_3_bias_v
identity_57��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*�
value�B�9B-final_dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB+final_dense/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBIfinal_dense/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBGfinal_dense/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBIfinal_dense/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBGfinal_dense/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:9*
dtype0*�
value|Bz9B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::*G
dtypes=
;29	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp)assignvariableop_wide_deep_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp)assignvariableop_1_wide_deep_dense_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_1Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_beta_2Identity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_adam_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp1assignvariableop_7_wide_deep_embedding_embeddingsIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp3assignvariableop_8_wide_deep_embedding_1_embeddingsIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp3assignvariableop_9_wide_deep_embedding_2_embeddingsIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp4assignvariableop_10_wide_deep_embedding_3_embeddingsIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp4assignvariableop_11_wide_deep_embedding_4_embeddingsIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp.assignvariableop_12_wide_deep_dnn_dense_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp,assignvariableop_13_wide_deep_dnn_dense_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp0assignvariableop_14_wide_deep_dnn_dense_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp.assignvariableop_15_wide_deep_dnn_dense_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp0assignvariableop_16_wide_deep_dnn_dense_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp.assignvariableop_17_wide_deep_dnn_dense_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp3assignvariableop_18_wide_deep_linear_dense_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp1assignvariableop_19_wide_deep_linear_dense_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp"assignvariableop_22_true_positivesIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp"assignvariableop_23_true_negativesIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp#assignvariableop_24_false_positivesIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp#assignvariableop_25_false_negativesIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp3assignvariableop_26_adam_wide_deep_dense_4_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp1assignvariableop_27_adam_wide_deep_dense_4_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp9assignvariableop_28_adam_wide_deep_embedding_embeddings_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp;assignvariableop_29_adam_wide_deep_embedding_1_embeddings_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp;assignvariableop_30_adam_wide_deep_embedding_2_embeddings_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp;assignvariableop_31_adam_wide_deep_embedding_3_embeddings_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp;assignvariableop_32_adam_wide_deep_embedding_4_embeddings_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp5assignvariableop_33_adam_wide_deep_dnn_dense_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp3assignvariableop_34_adam_wide_deep_dnn_dense_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp7assignvariableop_35_adam_wide_deep_dnn_dense_1_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_wide_deep_dnn_dense_1_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_wide_deep_dnn_dense_2_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp5assignvariableop_38_adam_wide_deep_dnn_dense_2_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp:assignvariableop_39_adam_wide_deep_linear_dense_3_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp8assignvariableop_40_adam_wide_deep_linear_dense_3_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp3assignvariableop_41_adam_wide_deep_dense_4_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp1assignvariableop_42_adam_wide_deep_dense_4_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp9assignvariableop_43_adam_wide_deep_embedding_embeddings_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp;assignvariableop_44_adam_wide_deep_embedding_1_embeddings_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp;assignvariableop_45_adam_wide_deep_embedding_2_embeddings_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp;assignvariableop_46_adam_wide_deep_embedding_3_embeddings_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp;assignvariableop_47_adam_wide_deep_embedding_4_embeddings_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp5assignvariableop_48_adam_wide_deep_dnn_dense_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp3assignvariableop_49_adam_wide_deep_dnn_dense_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp7assignvariableop_50_adam_wide_deep_dnn_dense_1_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp5assignvariableop_51_adam_wide_deep_dnn_dense_1_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp7assignvariableop_52_adam_wide_deep_dnn_dense_2_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp5assignvariableop_53_adam_wide_deep_dnn_dense_2_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp:assignvariableop_54_adam_wide_deep_linear_dense_3_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp8assignvariableop_55_adam_wide_deep_linear_dense_3_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_559
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_56Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_56�

Identity_57IdentityIdentity_56:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_57"#
identity_57Identity_57:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
__inference_loss_fn_4_4251O
Kwide_deep_embedding_4_embeddings_regularizer_square_readvariableop_resource
identity��Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpKwide_deep_embedding_4_embeddings_regularizer_square_readvariableop_resource* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_4/embeddings/Regularizer/SquareSquareJwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_4/embeddings/Regularizer/Square�
2wide_deep/embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_4/embeddings/Regularizer/Const�
0wide_deep/embedding_4/embeddings/Regularizer/SumSum7wide_deep/embedding_4/embeddings/Regularizer/Square:y:0;wide_deep/embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/Sum�
2wide_deep/embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_4/embeddings/Regularizer/mul/x�
0wide_deep/embedding_4/embeddings/Regularizer/mulMul;wide_deep/embedding_4/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/mul�
IdentityIdentity4wide_deep/embedding_4/embeddings/Regularizer/mul:z:0C^wide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp
�

�
"__inference_signature_wrapper_3719
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__wrapped_model_28812
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*u
_input_shapesd
b:���������	:���������:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2
�

�
@__inference_linear_layer_call_and_return_conditional_losses_3055

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity��dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/BiasAdd�
IdentityIdentitydense_3/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������	::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�

�
@__inference_linear_layer_call_and_return_conditional_losses_4149

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity��dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/BiasAdd�
IdentityIdentitydense_3/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������	::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
��
�
C__inference_wide_deep_layer_call_and_return_conditional_losses_3364
input_1
input_2
embedding_3269
embedding_1_3276
embedding_2_3283
embedding_3_3290
embedding_4_3297
linear_3304
linear_3306
dnn_3309
dnn_3311
dnn_3313
dnn_3315
dnn_3317
dnn_3319
dense_4_3322
dense_4_3324
identity��dense_4/StatefulPartitionedCall�dnn/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�#embedding_3/StatefulPartitionedCall�#embedding_4/StatefulPartitionedCall�linear/StatefulPartitionedCall�@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinput_2strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice�
!embedding/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0embedding_3269*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_29052#
!embedding/StatefulPartitionedCall
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinput_2strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_1_3276*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_29362%
#embedding_1/StatefulPartitionedCall
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack�
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1�
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2�
strided_slice_2StridedSliceinput_2strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_2_3283*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_29672%
#embedding_2/StatefulPartitionedCall
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack�
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1�
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2�
strided_slice_3StridedSliceinput_2strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3�
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_3_3290*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_3_layer_call_and_return_conditional_losses_29982%
#embedding_3/StatefulPartitionedCall
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack�
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1�
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2�
strided_slice_4StridedSliceinput_2strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4�
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_4:output:0embedding_4_3297*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_4_layer_call_and_return_conditional_losses_30292%
#embedding_4/StatefulPartitionedCalle
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/axis�
concatConcatV2*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0,embedding_4/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������(2
concati
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat_1/axis�
concat_1ConcatV2concat:output:0input_1concat_1/axis:output:0*
N*
T0*'
_output_shapes
:���������12

concat_1�
linear/StatefulPartitionedCallStatefulPartitionedCallinput_1linear_3304linear_3306*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_linear_layer_call_and_return_conditional_losses_30652 
linear/StatefulPartitionedCall�
dnn/StatefulPartitionedCallStatefulPartitionedCallconcat_1:output:0dnn_3309dnn_3311dnn_3313dnn_3315dnn_3317dnn_3319*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_dnn_layer_call_and_return_conditional_losses_31492
dnn/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall$dnn/StatefulPartitionedCall:output:0dense_4_3322dense_4_3324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_32082!
dense_4/StatefulPartitionedCallS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x|
mulMulmul/x:output:0'linear/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/x�
mul_1Mulmul_1/x:output:0(dense_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
mul_1Y
addAddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_3269* 
_output_shapes
:
Ͷ*
dtype02B
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�
1wide_deep/embedding/embeddings/Regularizer/SquareSquareHwide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Ͷ23
1wide_deep/embedding/embeddings/Regularizer/Square�
0wide_deep/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       22
0wide_deep/embedding/embeddings/Regularizer/Const�
.wide_deep/embedding/embeddings/Regularizer/SumSum5wide_deep/embedding/embeddings/Regularizer/Square:y:09wide_deep/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/Sum�
0wide_deep/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��822
0wide_deep/embedding/embeddings/Regularizer/mul/x�
.wide_deep/embedding/embeddings/Regularizer/mulMul9wide_deep/embedding/embeddings/Regularizer/mul/x:output:07wide_deep/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/mul�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_3276* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_1/embeddings/Regularizer/SquareSquareJwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_1/embeddings/Regularizer/Square�
2wide_deep/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_1/embeddings/Regularizer/Const�
0wide_deep/embedding_1/embeddings/Regularizer/SumSum7wide_deep/embedding_1/embeddings/Regularizer/Square:y:0;wide_deep/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/Sum�
2wide_deep/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_1/embeddings/Regularizer/mul/x�
0wide_deep/embedding_1/embeddings/Regularizer/mulMul;wide_deep/embedding_1/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/mul�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_3283*
_output_shapes
:	�*
dtype02D
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_2/embeddings/Regularizer/SquareSquareJwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�25
3wide_deep/embedding_2/embeddings/Regularizer/Square�
2wide_deep/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_2/embeddings/Regularizer/Const�
0wide_deep/embedding_2/embeddings/Regularizer/SumSum7wide_deep/embedding_2/embeddings/Regularizer/Square:y:0;wide_deep/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/Sum�
2wide_deep/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_2/embeddings/Regularizer/mul/x�
0wide_deep/embedding_2/embeddings/Regularizer/mulMul;wide_deep/embedding_2/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/mul�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_3_3290*
_output_shapes

:*
dtype02D
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_3/embeddings/Regularizer/SquareSquareJwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:25
3wide_deep/embedding_3/embeddings/Regularizer/Square�
2wide_deep/embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_3/embeddings/Regularizer/Const�
0wide_deep/embedding_3/embeddings/Regularizer/SumSum7wide_deep/embedding_3/embeddings/Regularizer/Square:y:0;wide_deep/embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/Sum�
2wide_deep/embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_3/embeddings/Regularizer/mul/x�
0wide_deep/embedding_3/embeddings/Regularizer/mulMul;wide_deep/embedding_3/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/mul�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_4_3297* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_4/embeddings/Regularizer/SquareSquareJwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_4/embeddings/Regularizer/Square�
2wide_deep/embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_4/embeddings/Regularizer/Const�
0wide_deep/embedding_4/embeddings/Regularizer/SumSum7wide_deep/embedding_4/embeddings/Regularizer/Square:y:0;wide_deep/embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/Sum�
2wide_deep/embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_4/embeddings/Regularizer/mul/x�
0wide_deep/embedding_4/embeddings/Regularizer/mulMul;wide_deep/embedding_4/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/mul�
IdentityIdentitySigmoid:y:0 ^dense_4/StatefulPartitionedCall^dnn/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall^linear/StatefulPartitionedCallA^wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*u
_input_shapesd
b:���������	:���������:::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2:
dnn/StatefulPartitionedCalldnn/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2@
linear/StatefulPartitionedCalllinear/StatefulPartitionedCall2�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
�
(__inference_wide_deep_layer_call_fn_4046
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_wide_deep_layer_call_and_return_conditional_losses_36102
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*u
_input_shapesd
b:���������	:���������:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:���������	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
E__inference_embedding_1_layer_call_and_return_conditional_losses_4300

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02!
embedding_lookup/ReadVariableOp�
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis�
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:���������2
embedding_lookup/Identity�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_1/embeddings/Regularizer/SquareSquareJwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_1/embeddings/Regularizer/Square�
2wide_deep/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_1/embeddings/Regularizer/Const�
0wide_deep/embedding_1/embeddings/Regularizer/SumSum7wide_deep/embedding_1/embeddings/Regularizer/Square:y:0;wide_deep/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/Sum�
2wide_deep/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_1/embeddings/Regularizer/mul/x�
0wide_deep/embedding_1/embeddings/Regularizer/mulMul;wide_deep/embedding_1/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/mul�
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOpC^wide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp2�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
z
%__inference_linear_layer_call_fn_4168

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_linear_layer_call_and_return_conditional_losses_30552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������	::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
�
�
__inference__wrapped_model_2881
input_1
input_2@
<wide_deep_embedding_embedding_lookup_readvariableop_resourceB
>wide_deep_embedding_1_embedding_lookup_readvariableop_resourceB
>wide_deep_embedding_2_embedding_lookup_readvariableop_resourceB
>wide_deep_embedding_3_embedding_lookup_readvariableop_resourceB
>wide_deep_embedding_4_embedding_lookup_readvariableop_resource;
7wide_deep_linear_dense_3_matmul_readvariableop_resource<
8wide_deep_linear_dense_3_biasadd_readvariableop_resource6
2wide_deep_dnn_dense_matmul_readvariableop_resource7
3wide_deep_dnn_dense_biasadd_readvariableop_resource8
4wide_deep_dnn_dense_1_matmul_readvariableop_resource9
5wide_deep_dnn_dense_1_biasadd_readvariableop_resource8
4wide_deep_dnn_dense_2_matmul_readvariableop_resource9
5wide_deep_dnn_dense_2_biasadd_readvariableop_resource4
0wide_deep_dense_4_matmul_readvariableop_resource5
1wide_deep_dense_4_biasadd_readvariableop_resource
identity��(wide_deep/dense_4/BiasAdd/ReadVariableOp�'wide_deep/dense_4/MatMul/ReadVariableOp�*wide_deep/dnn/dense/BiasAdd/ReadVariableOp�)wide_deep/dnn/dense/MatMul/ReadVariableOp�,wide_deep/dnn/dense_1/BiasAdd/ReadVariableOp�+wide_deep/dnn/dense_1/MatMul/ReadVariableOp�,wide_deep/dnn/dense_2/BiasAdd/ReadVariableOp�+wide_deep/dnn/dense_2/MatMul/ReadVariableOp�3wide_deep/embedding/embedding_lookup/ReadVariableOp�5wide_deep/embedding_1/embedding_lookup/ReadVariableOp�5wide_deep/embedding_2/embedding_lookup/ReadVariableOp�5wide_deep/embedding_3/embedding_lookup/ReadVariableOp�5wide_deep/embedding_4/embedding_lookup/ReadVariableOp�/wide_deep/linear/dense_3/BiasAdd/ReadVariableOp�.wide_deep/linear/dense_3/MatMul/ReadVariableOp�
wide_deep/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
wide_deep/strided_slice/stack�
wide_deep/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2!
wide_deep/strided_slice/stack_1�
wide_deep/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
wide_deep/strided_slice/stack_2�
wide_deep/strided_sliceStridedSliceinput_2&wide_deep/strided_slice/stack:output:0(wide_deep/strided_slice/stack_1:output:0(wide_deep/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
wide_deep/strided_slice�
3wide_deep/embedding/embedding_lookup/ReadVariableOpReadVariableOp<wide_deep_embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
Ͷ*
dtype025
3wide_deep/embedding/embedding_lookup/ReadVariableOp�
)wide_deep/embedding/embedding_lookup/axisConst*F
_class<
:8loc:@wide_deep/embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2+
)wide_deep/embedding/embedding_lookup/axis�
$wide_deep/embedding/embedding_lookupGatherV2;wide_deep/embedding/embedding_lookup/ReadVariableOp:value:0 wide_deep/strided_slice:output:02wide_deep/embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*F
_class<
:8loc:@wide_deep/embedding/embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2&
$wide_deep/embedding/embedding_lookup�
-wide_deep/embedding/embedding_lookup/IdentityIdentity-wide_deep/embedding/embedding_lookup:output:0*
T0*'
_output_shapes
:���������2/
-wide_deep/embedding/embedding_lookup/Identity�
wide_deep/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
wide_deep/strided_slice_1/stack�
!wide_deep/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!wide_deep/strided_slice_1/stack_1�
!wide_deep/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!wide_deep/strided_slice_1/stack_2�
wide_deep/strided_slice_1StridedSliceinput_2(wide_deep/strided_slice_1/stack:output:0*wide_deep/strided_slice_1/stack_1:output:0*wide_deep/strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
wide_deep/strided_slice_1�
5wide_deep/embedding_1/embedding_lookup/ReadVariableOpReadVariableOp>wide_deep_embedding_1_embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype027
5wide_deep/embedding_1/embedding_lookup/ReadVariableOp�
+wide_deep/embedding_1/embedding_lookup/axisConst*H
_class>
<:loc:@wide_deep/embedding_1/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2-
+wide_deep/embedding_1/embedding_lookup/axis�
&wide_deep/embedding_1/embedding_lookupGatherV2=wide_deep/embedding_1/embedding_lookup/ReadVariableOp:value:0"wide_deep/strided_slice_1:output:04wide_deep/embedding_1/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*H
_class>
<:loc:@wide_deep/embedding_1/embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2(
&wide_deep/embedding_1/embedding_lookup�
/wide_deep/embedding_1/embedding_lookup/IdentityIdentity/wide_deep/embedding_1/embedding_lookup:output:0*
T0*'
_output_shapes
:���������21
/wide_deep/embedding_1/embedding_lookup/Identity�
wide_deep/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
wide_deep/strided_slice_2/stack�
!wide_deep/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!wide_deep/strided_slice_2/stack_1�
!wide_deep/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!wide_deep/strided_slice_2/stack_2�
wide_deep/strided_slice_2StridedSliceinput_2(wide_deep/strided_slice_2/stack:output:0*wide_deep/strided_slice_2/stack_1:output:0*wide_deep/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
wide_deep/strided_slice_2�
5wide_deep/embedding_2/embedding_lookup/ReadVariableOpReadVariableOp>wide_deep_embedding_2_embedding_lookup_readvariableop_resource*
_output_shapes
:	�*
dtype027
5wide_deep/embedding_2/embedding_lookup/ReadVariableOp�
+wide_deep/embedding_2/embedding_lookup/axisConst*H
_class>
<:loc:@wide_deep/embedding_2/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2-
+wide_deep/embedding_2/embedding_lookup/axis�
&wide_deep/embedding_2/embedding_lookupGatherV2=wide_deep/embedding_2/embedding_lookup/ReadVariableOp:value:0"wide_deep/strided_slice_2:output:04wide_deep/embedding_2/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*H
_class>
<:loc:@wide_deep/embedding_2/embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2(
&wide_deep/embedding_2/embedding_lookup�
/wide_deep/embedding_2/embedding_lookup/IdentityIdentity/wide_deep/embedding_2/embedding_lookup:output:0*
T0*'
_output_shapes
:���������21
/wide_deep/embedding_2/embedding_lookup/Identity�
wide_deep/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
wide_deep/strided_slice_3/stack�
!wide_deep/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!wide_deep/strided_slice_3/stack_1�
!wide_deep/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!wide_deep/strided_slice_3/stack_2�
wide_deep/strided_slice_3StridedSliceinput_2(wide_deep/strided_slice_3/stack:output:0*wide_deep/strided_slice_3/stack_1:output:0*wide_deep/strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
wide_deep/strided_slice_3�
5wide_deep/embedding_3/embedding_lookup/ReadVariableOpReadVariableOp>wide_deep_embedding_3_embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype027
5wide_deep/embedding_3/embedding_lookup/ReadVariableOp�
+wide_deep/embedding_3/embedding_lookup/axisConst*H
_class>
<:loc:@wide_deep/embedding_3/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2-
+wide_deep/embedding_3/embedding_lookup/axis�
&wide_deep/embedding_3/embedding_lookupGatherV2=wide_deep/embedding_3/embedding_lookup/ReadVariableOp:value:0"wide_deep/strided_slice_3:output:04wide_deep/embedding_3/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*H
_class>
<:loc:@wide_deep/embedding_3/embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2(
&wide_deep/embedding_3/embedding_lookup�
/wide_deep/embedding_3/embedding_lookup/IdentityIdentity/wide_deep/embedding_3/embedding_lookup:output:0*
T0*'
_output_shapes
:���������21
/wide_deep/embedding_3/embedding_lookup/Identity�
wide_deep/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2!
wide_deep/strided_slice_4/stack�
!wide_deep/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!wide_deep/strided_slice_4/stack_1�
!wide_deep/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!wide_deep/strided_slice_4/stack_2�
wide_deep/strided_slice_4StridedSliceinput_2(wide_deep/strided_slice_4/stack:output:0*wide_deep/strided_slice_4/stack_1:output:0*wide_deep/strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
wide_deep/strided_slice_4�
5wide_deep/embedding_4/embedding_lookup/ReadVariableOpReadVariableOp>wide_deep_embedding_4_embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype027
5wide_deep/embedding_4/embedding_lookup/ReadVariableOp�
+wide_deep/embedding_4/embedding_lookup/axisConst*H
_class>
<:loc:@wide_deep/embedding_4/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2-
+wide_deep/embedding_4/embedding_lookup/axis�
&wide_deep/embedding_4/embedding_lookupGatherV2=wide_deep/embedding_4/embedding_lookup/ReadVariableOp:value:0"wide_deep/strided_slice_4:output:04wide_deep/embedding_4/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*H
_class>
<:loc:@wide_deep/embedding_4/embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2(
&wide_deep/embedding_4/embedding_lookup�
/wide_deep/embedding_4/embedding_lookup/IdentityIdentity/wide_deep/embedding_4/embedding_lookup:output:0*
T0*'
_output_shapes
:���������21
/wide_deep/embedding_4/embedding_lookup/Identityy
wide_deep/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
wide_deep/concat/axis�
wide_deep/concatConcatV26wide_deep/embedding/embedding_lookup/Identity:output:08wide_deep/embedding_1/embedding_lookup/Identity:output:08wide_deep/embedding_2/embedding_lookup/Identity:output:08wide_deep/embedding_3/embedding_lookup/Identity:output:08wide_deep/embedding_4/embedding_lookup/Identity:output:0wide_deep/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������(2
wide_deep/concat}
wide_deep/concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
wide_deep/concat_1/axis�
wide_deep/concat_1ConcatV2wide_deep/concat:output:0input_1 wide_deep/concat_1/axis:output:0*
N*
T0*'
_output_shapes
:���������12
wide_deep/concat_1�
.wide_deep/linear/dense_3/MatMul/ReadVariableOpReadVariableOp7wide_deep_linear_dense_3_matmul_readvariableop_resource*
_output_shapes

:	*
dtype020
.wide_deep/linear/dense_3/MatMul/ReadVariableOp�
wide_deep/linear/dense_3/MatMulMatMulinput_16wide_deep/linear/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2!
wide_deep/linear/dense_3/MatMul�
/wide_deep/linear/dense_3/BiasAdd/ReadVariableOpReadVariableOp8wide_deep_linear_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/wide_deep/linear/dense_3/BiasAdd/ReadVariableOp�
 wide_deep/linear/dense_3/BiasAddBiasAdd)wide_deep/linear/dense_3/MatMul:product:07wide_deep/linear/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2"
 wide_deep/linear/dense_3/BiasAdd�
)wide_deep/dnn/dense/MatMul/ReadVariableOpReadVariableOp2wide_deep_dnn_dense_matmul_readvariableop_resource*
_output_shapes
:	1�*
dtype02+
)wide_deep/dnn/dense/MatMul/ReadVariableOp�
wide_deep/dnn/dense/MatMulMatMulwide_deep/concat_1:output:01wide_deep/dnn/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
wide_deep/dnn/dense/MatMul�
*wide_deep/dnn/dense/BiasAdd/ReadVariableOpReadVariableOp3wide_deep_dnn_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02,
*wide_deep/dnn/dense/BiasAdd/ReadVariableOp�
wide_deep/dnn/dense/BiasAddBiasAdd$wide_deep/dnn/dense/MatMul:product:02wide_deep/dnn/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
wide_deep/dnn/dense/BiasAdd�
wide_deep/dnn/dense/ReluRelu$wide_deep/dnn/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
wide_deep/dnn/dense/Relu�
+wide_deep/dnn/dense_1/MatMul/ReadVariableOpReadVariableOp4wide_deep_dnn_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+wide_deep/dnn/dense_1/MatMul/ReadVariableOp�
wide_deep/dnn/dense_1/MatMulMatMul&wide_deep/dnn/dense/Relu:activations:03wide_deep/dnn/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
wide_deep/dnn/dense_1/MatMul�
,wide_deep/dnn/dense_1/BiasAdd/ReadVariableOpReadVariableOp5wide_deep_dnn_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02.
,wide_deep/dnn/dense_1/BiasAdd/ReadVariableOp�
wide_deep/dnn/dense_1/BiasAddBiasAdd&wide_deep/dnn/dense_1/MatMul:product:04wide_deep/dnn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
wide_deep/dnn/dense_1/BiasAdd�
wide_deep/dnn/dense_1/ReluRelu&wide_deep/dnn/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
wide_deep/dnn/dense_1/Relu�
+wide_deep/dnn/dense_2/MatMul/ReadVariableOpReadVariableOp4wide_deep_dnn_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02-
+wide_deep/dnn/dense_2/MatMul/ReadVariableOp�
wide_deep/dnn/dense_2/MatMulMatMul(wide_deep/dnn/dense_1/Relu:activations:03wide_deep/dnn/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
wide_deep/dnn/dense_2/MatMul�
,wide_deep/dnn/dense_2/BiasAdd/ReadVariableOpReadVariableOp5wide_deep_dnn_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,wide_deep/dnn/dense_2/BiasAdd/ReadVariableOp�
wide_deep/dnn/dense_2/BiasAddBiasAdd&wide_deep/dnn/dense_2/MatMul:product:04wide_deep/dnn/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
wide_deep/dnn/dense_2/BiasAdd�
wide_deep/dnn/dense_2/ReluRelu&wide_deep/dnn/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
wide_deep/dnn/dense_2/Relu�
wide_deep/dnn/dropout/IdentityIdentity(wide_deep/dnn/dense_2/Relu:activations:0*
T0*'
_output_shapes
:���������@2 
wide_deep/dnn/dropout/Identity�
'wide_deep/dense_4/MatMul/ReadVariableOpReadVariableOp0wide_deep_dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02)
'wide_deep/dense_4/MatMul/ReadVariableOp�
wide_deep/dense_4/MatMulMatMul'wide_deep/dnn/dropout/Identity:output:0/wide_deep/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
wide_deep/dense_4/MatMul�
(wide_deep/dense_4/BiasAdd/ReadVariableOpReadVariableOp1wide_deep_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(wide_deep/dense_4/BiasAdd/ReadVariableOp�
wide_deep/dense_4/BiasAddBiasAdd"wide_deep/dense_4/MatMul:product:00wide_deep/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
wide_deep/dense_4/BiasAddg
wide_deep/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
wide_deep/mul/x�
wide_deep/mulMulwide_deep/mul/x:output:0)wide_deep/linear/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
wide_deep/mulk
wide_deep/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
wide_deep/mul_1/x�
wide_deep/mul_1Mulwide_deep/mul_1/x:output:0"wide_deep/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
wide_deep/mul_1�
wide_deep/addAddV2wide_deep/mul:z:0wide_deep/mul_1:z:0*
T0*'
_output_shapes
:���������2
wide_deep/addv
wide_deep/SigmoidSigmoidwide_deep/add:z:0*
T0*'
_output_shapes
:���������2
wide_deep/Sigmoid�
IdentityIdentitywide_deep/Sigmoid:y:0)^wide_deep/dense_4/BiasAdd/ReadVariableOp(^wide_deep/dense_4/MatMul/ReadVariableOp+^wide_deep/dnn/dense/BiasAdd/ReadVariableOp*^wide_deep/dnn/dense/MatMul/ReadVariableOp-^wide_deep/dnn/dense_1/BiasAdd/ReadVariableOp,^wide_deep/dnn/dense_1/MatMul/ReadVariableOp-^wide_deep/dnn/dense_2/BiasAdd/ReadVariableOp,^wide_deep/dnn/dense_2/MatMul/ReadVariableOp4^wide_deep/embedding/embedding_lookup/ReadVariableOp6^wide_deep/embedding_1/embedding_lookup/ReadVariableOp6^wide_deep/embedding_2/embedding_lookup/ReadVariableOp6^wide_deep/embedding_3/embedding_lookup/ReadVariableOp6^wide_deep/embedding_4/embedding_lookup/ReadVariableOp0^wide_deep/linear/dense_3/BiasAdd/ReadVariableOp/^wide_deep/linear/dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*u
_input_shapesd
b:���������	:���������:::::::::::::::2T
(wide_deep/dense_4/BiasAdd/ReadVariableOp(wide_deep/dense_4/BiasAdd/ReadVariableOp2R
'wide_deep/dense_4/MatMul/ReadVariableOp'wide_deep/dense_4/MatMul/ReadVariableOp2X
*wide_deep/dnn/dense/BiasAdd/ReadVariableOp*wide_deep/dnn/dense/BiasAdd/ReadVariableOp2V
)wide_deep/dnn/dense/MatMul/ReadVariableOp)wide_deep/dnn/dense/MatMul/ReadVariableOp2\
,wide_deep/dnn/dense_1/BiasAdd/ReadVariableOp,wide_deep/dnn/dense_1/BiasAdd/ReadVariableOp2Z
+wide_deep/dnn/dense_1/MatMul/ReadVariableOp+wide_deep/dnn/dense_1/MatMul/ReadVariableOp2\
,wide_deep/dnn/dense_2/BiasAdd/ReadVariableOp,wide_deep/dnn/dense_2/BiasAdd/ReadVariableOp2Z
+wide_deep/dnn/dense_2/MatMul/ReadVariableOp+wide_deep/dnn/dense_2/MatMul/ReadVariableOp2j
3wide_deep/embedding/embedding_lookup/ReadVariableOp3wide_deep/embedding/embedding_lookup/ReadVariableOp2n
5wide_deep/embedding_1/embedding_lookup/ReadVariableOp5wide_deep/embedding_1/embedding_lookup/ReadVariableOp2n
5wide_deep/embedding_2/embedding_lookup/ReadVariableOp5wide_deep/embedding_2/embedding_lookup/ReadVariableOp2n
5wide_deep/embedding_3/embedding_lookup/ReadVariableOp5wide_deep/embedding_3/embedding_lookup/ReadVariableOp2n
5wide_deep/embedding_4/embedding_lookup/ReadVariableOp5wide_deep/embedding_4/embedding_lookup/ReadVariableOp2b
/wide_deep/linear/dense_3/BiasAdd/ReadVariableOp/wide_deep/linear/dense_3/BiasAdd/ReadVariableOp2`
.wide_deep/linear/dense_3/MatMul/ReadVariableOp.wide_deep/linear/dense_3/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
�
__inference_loss_fn_3_4240O
Kwide_deep_embedding_3_embeddings_regularizer_square_readvariableop_resource
identity��Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpKwide_deep_embedding_3_embeddings_regularizer_square_readvariableop_resource*
_output_shapes

:*
dtype02D
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_3/embeddings/Regularizer/SquareSquareJwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:25
3wide_deep/embedding_3/embeddings/Regularizer/Square�
2wide_deep/embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_3/embeddings/Regularizer/Const�
0wide_deep/embedding_3/embeddings/Regularizer/SumSum7wide_deep/embedding_3/embeddings/Regularizer/Square:y:0;wide_deep/embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/Sum�
2wide_deep/embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_3/embeddings/Regularizer/mul/x�
0wide_deep/embedding_3/embeddings/Regularizer/mulMul;wide_deep/embedding_3/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/mul�
IdentityIdentity4wide_deep/embedding_3/embeddings/Regularizer/mul:z:0C^wide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp
�

�
@__inference_linear_layer_call_and_return_conditional_losses_4159

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource
identity��dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_3/BiasAdd�
IdentityIdentitydense_3/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������	::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs
��
�
C__inference_wide_deep_layer_call_and_return_conditional_losses_3471

inputs
inputs_1
embedding_3376
embedding_1_3383
embedding_2_3390
embedding_3_3397
embedding_4_3404
linear_3411
linear_3413
dnn_3416
dnn_3418
dnn_3420
dnn_3422
dnn_3424
dnn_3426
dense_4_3429
dense_4_3431
identity��dense_4/StatefulPartitionedCall�dnn/StatefulPartitionedCall�!embedding/StatefulPartitionedCall�#embedding_1/StatefulPartitionedCall�#embedding_2/StatefulPartitionedCall�#embedding_3/StatefulPartitionedCall�#embedding_4/StatefulPartitionedCall�linear/StatefulPartitionedCall�@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinputs_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice�
!embedding/StatefulPartitionedCallStatefulPartitionedCallstrided_slice:output:0embedding_3376*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_29052#
!embedding/StatefulPartitionedCall
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputs_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1�
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0embedding_1_3383*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_29362%
#embedding_1/StatefulPartitionedCall
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack�
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1�
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2�
strided_slice_2StridedSliceinputs_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2�
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0embedding_2_3390*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_2_layer_call_and_return_conditional_losses_29672%
#embedding_2/StatefulPartitionedCall
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack�
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1�
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2�
strided_slice_3StridedSliceinputs_1strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3�
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_3:output:0embedding_3_3397*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_3_layer_call_and_return_conditional_losses_29982%
#embedding_3/StatefulPartitionedCall
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack�
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1�
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2�
strided_slice_4StridedSliceinputs_1strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4�
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_4:output:0embedding_4_3404*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_4_layer_call_and_return_conditional_losses_30292%
#embedding_4/StatefulPartitionedCalle
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/axis�
concatConcatV2*embedding/StatefulPartitionedCall:output:0,embedding_1/StatefulPartitionedCall:output:0,embedding_2/StatefulPartitionedCall:output:0,embedding_3/StatefulPartitionedCall:output:0,embedding_4/StatefulPartitionedCall:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������(2
concati
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat_1/axis�
concat_1ConcatV2concat:output:0inputsconcat_1/axis:output:0*
N*
T0*'
_output_shapes
:���������12

concat_1�
linear/StatefulPartitionedCallStatefulPartitionedCallinputslinear_3411linear_3413*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_linear_layer_call_and_return_conditional_losses_30552 
linear/StatefulPartitionedCall�
dnn/StatefulPartitionedCallStatefulPartitionedCallconcat_1:output:0dnn_3416dnn_3418dnn_3420dnn_3422dnn_3424dnn_3426*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *F
fAR?
=__inference_dnn_layer_call_and_return_conditional_losses_31232
dnn/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall$dnn/StatefulPartitionedCall:output:0dense_4_3429dense_4_3431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_32082!
dense_4/StatefulPartitionedCallS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/x|
mulMulmul/x:output:0'linear/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/x�
mul_1Mulmul_1/x:output:0(dense_4/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������2
mul_1Y
addAddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_3376* 
_output_shapes
:
Ͷ*
dtype02B
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�
1wide_deep/embedding/embeddings/Regularizer/SquareSquareHwide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Ͷ23
1wide_deep/embedding/embeddings/Regularizer/Square�
0wide_deep/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       22
0wide_deep/embedding/embeddings/Regularizer/Const�
.wide_deep/embedding/embeddings/Regularizer/SumSum5wide_deep/embedding/embeddings/Regularizer/Square:y:09wide_deep/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/Sum�
0wide_deep/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��822
0wide_deep/embedding/embeddings/Regularizer/mul/x�
.wide_deep/embedding/embeddings/Regularizer/mulMul9wide_deep/embedding/embeddings/Regularizer/mul/x:output:07wide_deep/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/mul�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_1_3383* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_1/embeddings/Regularizer/SquareSquareJwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_1/embeddings/Regularizer/Square�
2wide_deep/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_1/embeddings/Regularizer/Const�
0wide_deep/embedding_1/embeddings/Regularizer/SumSum7wide_deep/embedding_1/embeddings/Regularizer/Square:y:0;wide_deep/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/Sum�
2wide_deep/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_1/embeddings/Regularizer/mul/x�
0wide_deep/embedding_1/embeddings/Regularizer/mulMul;wide_deep/embedding_1/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/mul�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_2_3390*
_output_shapes
:	�*
dtype02D
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_2/embeddings/Regularizer/SquareSquareJwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�25
3wide_deep/embedding_2/embeddings/Regularizer/Square�
2wide_deep/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_2/embeddings/Regularizer/Const�
0wide_deep/embedding_2/embeddings/Regularizer/SumSum7wide_deep/embedding_2/embeddings/Regularizer/Square:y:0;wide_deep/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/Sum�
2wide_deep/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_2/embeddings/Regularizer/mul/x�
0wide_deep/embedding_2/embeddings/Regularizer/mulMul;wide_deep/embedding_2/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/mul�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_3_3397*
_output_shapes

:*
dtype02D
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_3/embeddings/Regularizer/SquareSquareJwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:25
3wide_deep/embedding_3/embeddings/Regularizer/Square�
2wide_deep/embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_3/embeddings/Regularizer/Const�
0wide_deep/embedding_3/embeddings/Regularizer/SumSum7wide_deep/embedding_3/embeddings/Regularizer/Square:y:0;wide_deep/embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/Sum�
2wide_deep/embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_3/embeddings/Regularizer/mul/x�
0wide_deep/embedding_3/embeddings/Regularizer/mulMul;wide_deep/embedding_3/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/mul�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_4_3404* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_4/embeddings/Regularizer/SquareSquareJwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_4/embeddings/Regularizer/Square�
2wide_deep/embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_4/embeddings/Regularizer/Const�
0wide_deep/embedding_4/embeddings/Regularizer/SumSum7wide_deep/embedding_4/embeddings/Regularizer/Square:y:0;wide_deep/embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/Sum�
2wide_deep/embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_4/embeddings/Regularizer/mul/x�
0wide_deep/embedding_4/embeddings/Regularizer/mulMul;wide_deep/embedding_4/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/mul�
IdentityIdentitySigmoid:y:0 ^dense_4/StatefulPartitionedCall^dnn/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall^linear/StatefulPartitionedCallA^wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*u
_input_shapesd
b:���������	:���������:::::::::::::::2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2:
dnn/StatefulPartitionedCalldnn/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2@
linear/StatefulPartitionedCalllinear/StatefulPartitionedCall2�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:���������	
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
{
&__inference_dense_4_layer_call_fn_4196

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_dense_4_layer_call_and_return_conditional_losses_32082
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
E__inference_embedding_2_layer_call_and_return_conditional_losses_4328

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	�*
dtype02!
embedding_lookup/ReadVariableOp�
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis�
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:���������2
embedding_lookup/Identity�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	�*
dtype02D
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_2/embeddings/Regularizer/SquareSquareJwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�25
3wide_deep/embedding_2/embeddings/Regularizer/Square�
2wide_deep/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_2/embeddings/Regularizer/Const�
0wide_deep/embedding_2/embeddings/Regularizer/SumSum7wide_deep/embedding_2/embeddings/Regularizer/Square:y:0;wide_deep/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/Sum�
2wide_deep/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_2/embeddings/Regularizer/mul/x�
0wide_deep/embedding_2/embeddings/Regularizer/mulMul;wide_deep/embedding_2/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/mul�
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOpC^wide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp2�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
p
*__inference_embedding_1_layer_call_fn_4307

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_embedding_1_layer_call_and_return_conditional_losses_29362
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
��
�
C__inference_wide_deep_layer_call_and_return_conditional_losses_3850
inputs_0
inputs_16
2embedding_embedding_lookup_readvariableop_resource8
4embedding_1_embedding_lookup_readvariableop_resource8
4embedding_2_embedding_lookup_readvariableop_resource8
4embedding_3_embedding_lookup_readvariableop_resource8
4embedding_4_embedding_lookup_readvariableop_resource1
-linear_dense_3_matmul_readvariableop_resource2
.linear_dense_3_biasadd_readvariableop_resource,
(dnn_dense_matmul_readvariableop_resource-
)dnn_dense_biasadd_readvariableop_resource.
*dnn_dense_1_matmul_readvariableop_resource/
+dnn_dense_1_biasadd_readvariableop_resource.
*dnn_dense_2_matmul_readvariableop_resource/
+dnn_dense_2_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource
identity��dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp� dnn/dense/BiasAdd/ReadVariableOp�dnn/dense/MatMul/ReadVariableOp�"dnn/dense_1/BiasAdd/ReadVariableOp�!dnn/dense_1/MatMul/ReadVariableOp�"dnn/dense_2/BiasAdd/ReadVariableOp�!dnn/dense_2/MatMul/ReadVariableOp�)embedding/embedding_lookup/ReadVariableOp�+embedding_1/embedding_lookup/ReadVariableOp�+embedding_2/embedding_lookup/ReadVariableOp�+embedding_3/embedding_lookup/ReadVariableOp�+embedding_4/embedding_lookup/ReadVariableOp�%linear/dense_3/BiasAdd/ReadVariableOp�$linear/dense_3/MatMul/ReadVariableOp�@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2�
strided_sliceStridedSliceinputs_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice�
)embedding/embedding_lookup/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
Ͷ*
dtype02+
)embedding/embedding_lookup/ReadVariableOp�
embedding/embedding_lookup/axisConst*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2!
embedding/embedding_lookup/axis�
embedding/embedding_lookupGatherV21embedding/embedding_lookup/ReadVariableOp:value:0strided_slice:output:0(embedding/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*<
_class2
0.loc:@embedding/embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding/embedding_lookup�
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*'
_output_shapes
:���������2%
#embedding/embedding_lookup/Identity
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack�
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_1/stack_1�
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2�
strided_slice_1StridedSliceinputs_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1�
+embedding_1/embedding_lookup/ReadVariableOpReadVariableOp4embedding_1_embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+embedding_1/embedding_lookup/ReadVariableOp�
!embedding_1/embedding_lookup/axisConst*>
_class4
20loc:@embedding_1/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2#
!embedding_1/embedding_lookup/axis�
embedding_1/embedding_lookupGatherV23embedding_1/embedding_lookup/ReadVariableOp:value:0strided_slice_1:output:0*embedding_1/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*>
_class4
20loc:@embedding_1/embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_1/embedding_lookup�
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*'
_output_shapes
:���������2'
%embedding_1/embedding_lookup/Identity
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack�
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1�
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2�
strided_slice_2StridedSliceinputs_1strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2�
+embedding_2/embedding_lookup/ReadVariableOpReadVariableOp4embedding_2_embedding_lookup_readvariableop_resource*
_output_shapes
:	�*
dtype02-
+embedding_2/embedding_lookup/ReadVariableOp�
!embedding_2/embedding_lookup/axisConst*>
_class4
20loc:@embedding_2/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2#
!embedding_2/embedding_lookup/axis�
embedding_2/embedding_lookupGatherV23embedding_2/embedding_lookup/ReadVariableOp:value:0strided_slice_2:output:0*embedding_2/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*>
_class4
20loc:@embedding_2/embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_2/embedding_lookup�
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*'
_output_shapes
:���������2'
%embedding_2/embedding_lookup/Identity
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack�
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_3/stack_1�
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2�
strided_slice_3StridedSliceinputs_1strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_3�
+embedding_3/embedding_lookup/ReadVariableOpReadVariableOp4embedding_3_embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype02-
+embedding_3/embedding_lookup/ReadVariableOp�
!embedding_3/embedding_lookup/axisConst*>
_class4
20loc:@embedding_3/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2#
!embedding_3/embedding_lookup/axis�
embedding_3/embedding_lookupGatherV23embedding_3/embedding_lookup/ReadVariableOp:value:0strided_slice_3:output:0*embedding_3/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*>
_class4
20loc:@embedding_3/embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_3/embedding_lookup�
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0*
T0*'
_output_shapes
:���������2'
%embedding_3/embedding_lookup/Identity
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack�
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1�
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2�
strided_slice_4StridedSliceinputs_1strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_4�
+embedding_4/embedding_lookup/ReadVariableOpReadVariableOp4embedding_4_embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02-
+embedding_4/embedding_lookup/ReadVariableOp�
!embedding_4/embedding_lookup/axisConst*>
_class4
20loc:@embedding_4/embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2#
!embedding_4/embedding_lookup/axis�
embedding_4/embedding_lookupGatherV23embedding_4/embedding_lookup/ReadVariableOp:value:0strided_slice_4:output:0*embedding_4/embedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*>
_class4
20loc:@embedding_4/embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_4/embedding_lookup�
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0*
T0*'
_output_shapes
:���������2'
%embedding_4/embedding_lookup/Identitye
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat/axis�
concatConcatV2,embedding/embedding_lookup/Identity:output:0.embedding_1/embedding_lookup/Identity:output:0.embedding_2/embedding_lookup/Identity:output:0.embedding_3/embedding_lookup/Identity:output:0.embedding_4/embedding_lookup/Identity:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:���������(2
concati
concat_1/axisConst*
_output_shapes
: *
dtype0*
valueB :
���������2
concat_1/axis�
concat_1ConcatV2concat:output:0inputs_0concat_1/axis:output:0*
N*
T0*'
_output_shapes
:���������12

concat_1�
$linear/dense_3/MatMul/ReadVariableOpReadVariableOp-linear_dense_3_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02&
$linear/dense_3/MatMul/ReadVariableOp�
linear/dense_3/MatMulMatMulinputs_0,linear/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
linear/dense_3/MatMul�
%linear/dense_3/BiasAdd/ReadVariableOpReadVariableOp.linear_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%linear/dense_3/BiasAdd/ReadVariableOp�
linear/dense_3/BiasAddBiasAddlinear/dense_3/MatMul:product:0-linear/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
linear/dense_3/BiasAdd�
dnn/dense/MatMul/ReadVariableOpReadVariableOp(dnn_dense_matmul_readvariableop_resource*
_output_shapes
:	1�*
dtype02!
dnn/dense/MatMul/ReadVariableOp�
dnn/dense/MatMulMatMulconcat_1:output:0'dnn/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dnn/dense/MatMul�
 dnn/dense/BiasAdd/ReadVariableOpReadVariableOp)dnn_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02"
 dnn/dense/BiasAdd/ReadVariableOp�
dnn/dense/BiasAddBiasAdddnn/dense/MatMul:product:0(dnn/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dnn/dense/BiasAddw
dnn/dense/ReluReludnn/dense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dnn/dense/Relu�
!dnn/dense_1/MatMul/ReadVariableOpReadVariableOp*dnn_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02#
!dnn/dense_1/MatMul/ReadVariableOp�
dnn/dense_1/MatMulMatMuldnn/dense/Relu:activations:0)dnn/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dnn/dense_1/MatMul�
"dnn/dense_1/BiasAdd/ReadVariableOpReadVariableOp+dnn_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02$
"dnn/dense_1/BiasAdd/ReadVariableOp�
dnn/dense_1/BiasAddBiasAdddnn/dense_1/MatMul:product:0*dnn/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dnn/dense_1/BiasAdd}
dnn/dense_1/ReluReludnn/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dnn/dense_1/Relu�
!dnn/dense_2/MatMul/ReadVariableOpReadVariableOp*dnn_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02#
!dnn/dense_2/MatMul/ReadVariableOp�
dnn/dense_2/MatMulMatMuldnn/dense_1/Relu:activations:0)dnn/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dnn/dense_2/MatMul�
"dnn/dense_2/BiasAdd/ReadVariableOpReadVariableOp+dnn_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"dnn/dense_2/BiasAdd/ReadVariableOp�
dnn/dense_2/BiasAddBiasAdddnn/dense_2/MatMul:product:0*dnn/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dnn/dense_2/BiasAdd|
dnn/dense_2/ReluReludnn/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dnn/dense_2/Relu{
dnn/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?2
dnn/dropout/dropout/Const�
dnn/dropout/dropout/MulMuldnn/dense_2/Relu:activations:0"dnn/dropout/dropout/Const:output:0*
T0*'
_output_shapes
:���������@2
dnn/dropout/dropout/Mul�
dnn/dropout/dropout/ShapeShapednn/dense_2/Relu:activations:0*
T0*
_output_shapes
:2
dnn/dropout/dropout/Shape�
0dnn/dropout/dropout/random_uniform/RandomUniformRandomUniform"dnn/dropout/dropout/Shape:output:0*
T0*'
_output_shapes
:���������@*
dtype022
0dnn/dropout/dropout/random_uniform/RandomUniform�
"dnn/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"dnn/dropout/dropout/GreaterEqual/y�
 dnn/dropout/dropout/GreaterEqualGreaterEqual9dnn/dropout/dropout/random_uniform/RandomUniform:output:0+dnn/dropout/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:���������@2"
 dnn/dropout/dropout/GreaterEqual�
dnn/dropout/dropout/CastCast$dnn/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:���������@2
dnn/dropout/dropout/Cast�
dnn/dropout/dropout/Mul_1Muldnn/dropout/dropout/Mul:z:0dnn/dropout/dropout/Cast:y:0*
T0*'
_output_shapes
:���������@2
dnn/dropout/dropout/Mul_1�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMuldnn/dropout/dropout/Mul_1:z:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_4/BiasAddS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
mul/xt
mulMulmul/x:output:0linear/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
mulW
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/xs
mul_1Mulmul_1/x:output:0dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:���������2
mul_1Y
addAddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:���������2
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:���������2	
Sigmoid�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp2embedding_embedding_lookup_readvariableop_resource* 
_output_shapes
:
Ͷ*
dtype02B
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�
1wide_deep/embedding/embeddings/Regularizer/SquareSquareHwide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Ͷ23
1wide_deep/embedding/embeddings/Regularizer/Square�
0wide_deep/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       22
0wide_deep/embedding/embeddings/Regularizer/Const�
.wide_deep/embedding/embeddings/Regularizer/SumSum5wide_deep/embedding/embeddings/Regularizer/Square:y:09wide_deep/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/Sum�
0wide_deep/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��822
0wide_deep/embedding/embeddings/Regularizer/mul/x�
.wide_deep/embedding/embeddings/Regularizer/mulMul9wide_deep/embedding/embeddings/Regularizer/mul/x:output:07wide_deep/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/mul�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp4embedding_1_embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_1/embeddings/Regularizer/SquareSquareJwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_1/embeddings/Regularizer/Square�
2wide_deep/embedding_1/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_1/embeddings/Regularizer/Const�
0wide_deep/embedding_1/embeddings/Regularizer/SumSum7wide_deep/embedding_1/embeddings/Regularizer/Square:y:0;wide_deep/embedding_1/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/Sum�
2wide_deep/embedding_1/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_1/embeddings/Regularizer/mul/x�
0wide_deep/embedding_1/embeddings/Regularizer/mulMul;wide_deep/embedding_1/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_1/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_1/embeddings/Regularizer/mul�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp4embedding_2_embedding_lookup_readvariableop_resource*
_output_shapes
:	�*
dtype02D
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_2/embeddings/Regularizer/SquareSquareJwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�25
3wide_deep/embedding_2/embeddings/Regularizer/Square�
2wide_deep/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_2/embeddings/Regularizer/Const�
0wide_deep/embedding_2/embeddings/Regularizer/SumSum7wide_deep/embedding_2/embeddings/Regularizer/Square:y:0;wide_deep/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/Sum�
2wide_deep/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_2/embeddings/Regularizer/mul/x�
0wide_deep/embedding_2/embeddings/Regularizer/mulMul;wide_deep/embedding_2/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/mul�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp4embedding_3_embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype02D
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_3/embeddings/Regularizer/SquareSquareJwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:25
3wide_deep/embedding_3/embeddings/Regularizer/Square�
2wide_deep/embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_3/embeddings/Regularizer/Const�
0wide_deep/embedding_3/embeddings/Regularizer/SumSum7wide_deep/embedding_3/embeddings/Regularizer/Square:y:0;wide_deep/embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/Sum�
2wide_deep/embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_3/embeddings/Regularizer/mul/x�
0wide_deep/embedding_3/embeddings/Regularizer/mulMul;wide_deep/embedding_3/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/mul�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp4embedding_4_embedding_lookup_readvariableop_resource* 
_output_shapes
:
��*
dtype02D
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_4/embeddings/Regularizer/SquareSquareJwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
��25
3wide_deep/embedding_4/embeddings/Regularizer/Square�
2wide_deep/embedding_4/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_4/embeddings/Regularizer/Const�
0wide_deep/embedding_4/embeddings/Regularizer/SumSum7wide_deep/embedding_4/embeddings/Regularizer/Square:y:0;wide_deep/embedding_4/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/Sum�
2wide_deep/embedding_4/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_4/embeddings/Regularizer/mul/x�
0wide_deep/embedding_4/embeddings/Regularizer/mulMul;wide_deep/embedding_4/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_4/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_4/embeddings/Regularizer/mul�
IdentityIdentitySigmoid:y:0^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp!^dnn/dense/BiasAdd/ReadVariableOp ^dnn/dense/MatMul/ReadVariableOp#^dnn/dense_1/BiasAdd/ReadVariableOp"^dnn/dense_1/MatMul/ReadVariableOp#^dnn/dense_2/BiasAdd/ReadVariableOp"^dnn/dense_2/MatMul/ReadVariableOp*^embedding/embedding_lookup/ReadVariableOp,^embedding_1/embedding_lookup/ReadVariableOp,^embedding_2/embedding_lookup/ReadVariableOp,^embedding_3/embedding_lookup/ReadVariableOp,^embedding_4/embedding_lookup/ReadVariableOp&^linear/dense_3/BiasAdd/ReadVariableOp%^linear/dense_3/MatMul/ReadVariableOpA^wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpC^wide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*u
_input_shapesd
b:���������	:���������:::::::::::::::2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2D
 dnn/dense/BiasAdd/ReadVariableOp dnn/dense/BiasAdd/ReadVariableOp2B
dnn/dense/MatMul/ReadVariableOpdnn/dense/MatMul/ReadVariableOp2H
"dnn/dense_1/BiasAdd/ReadVariableOp"dnn/dense_1/BiasAdd/ReadVariableOp2F
!dnn/dense_1/MatMul/ReadVariableOp!dnn/dense_1/MatMul/ReadVariableOp2H
"dnn/dense_2/BiasAdd/ReadVariableOp"dnn/dense_2/BiasAdd/ReadVariableOp2F
!dnn/dense_2/MatMul/ReadVariableOp!dnn/dense_2/MatMul/ReadVariableOp2V
)embedding/embedding_lookup/ReadVariableOp)embedding/embedding_lookup/ReadVariableOp2Z
+embedding_1/embedding_lookup/ReadVariableOp+embedding_1/embedding_lookup/ReadVariableOp2Z
+embedding_2/embedding_lookup/ReadVariableOp+embedding_2/embedding_lookup/ReadVariableOp2Z
+embedding_3/embedding_lookup/ReadVariableOp+embedding_3/embedding_lookup/ReadVariableOp2Z
+embedding_4/embedding_lookup/ReadVariableOp+embedding_4/embedding_lookup/ReadVariableOp2N
%linear/dense_3/BiasAdd/ReadVariableOp%linear/dense_3/BiasAdd/ReadVariableOp2L
$linear/dense_3/MatMul/ReadVariableOp$linear/dense_3/MatMul/ReadVariableOp2�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_1/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp2�
Bwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_4/embeddings/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:���������	
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�	
�
A__inference_dense_4_layer_call_and_return_conditional_losses_3208

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
C__inference_embedding_layer_call_and_return_conditional_losses_4272

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
Ͷ*
dtype02!
embedding_lookup/ReadVariableOp�
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis�
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:���������2
embedding_lookup/Identity�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
Ͷ*
dtype02B
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�
1wide_deep/embedding/embeddings/Regularizer/SquareSquareHwide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Ͷ23
1wide_deep/embedding/embeddings/Regularizer/Square�
0wide_deep/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       22
0wide_deep/embedding/embeddings/Regularizer/Const�
.wide_deep/embedding/embeddings/Regularizer/SumSum5wide_deep/embedding/embeddings/Regularizer/Square:y:09wide_deep/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/Sum�
0wide_deep/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��822
0wide_deep/embedding/embeddings/Regularizer/mul/x�
.wide_deep/embedding/embeddings/Regularizer/mulMul9wide_deep/embedding/embeddings/Regularizer/mul/x:output:07wide_deep/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/mul�
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOpA^wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp2�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_wide_deep_layer_call_fn_3504
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*1
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_wide_deep_layer_call_and_return_conditional_losses_34712
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*u
_input_shapesd
b:���������	:���������:::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������	
!
_user_specified_name	input_1:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
�
C__inference_embedding_layer_call_and_return_conditional_losses_2905

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
Ͷ*
dtype02!
embedding_lookup/ReadVariableOp�
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis�
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:���������2
embedding_lookup/Identity�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource* 
_output_shapes
:
Ͷ*
dtype02B
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp�
1wide_deep/embedding/embeddings/Regularizer/SquareSquareHwide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
Ͷ23
1wide_deep/embedding/embeddings/Regularizer/Square�
0wide_deep/embedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       22
0wide_deep/embedding/embeddings/Regularizer/Const�
.wide_deep/embedding/embeddings/Regularizer/SumSum5wide_deep/embedding/embeddings/Regularizer/Square:y:09wide_deep/embedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/Sum�
0wide_deep/embedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��822
0wide_deep/embedding/embeddings/Regularizer/mul/x�
.wide_deep/embedding/embeddings/Regularizer/mulMul9wide_deep/embedding/embeddings/Regularizer/mul/x:output:07wide_deep/embedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 20
.wide_deep/embedding/embeddings/Regularizer/mul�
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOpA^wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp2�
@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp@wide_deep/embedding/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
E__inference_embedding_2_layer_call_and_return_conditional_losses_2967

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	�*
dtype02!
embedding_lookup/ReadVariableOp�
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis�
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:���������2
embedding_lookup/Identity�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes
:	�*
dtype02D
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_2/embeddings/Regularizer/SquareSquareJwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	�25
3wide_deep/embedding_2/embeddings/Regularizer/Square�
2wide_deep/embedding_2/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_2/embeddings/Regularizer/Const�
0wide_deep/embedding_2/embeddings/Regularizer/SumSum7wide_deep/embedding_2/embeddings/Regularizer/Square:y:0;wide_deep/embedding_2/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/Sum�
2wide_deep/embedding_2/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_2/embeddings/Regularizer/mul/x�
0wide_deep/embedding_2/embeddings/Regularizer/mulMul;wide_deep/embedding_2/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_2/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_2/embeddings/Regularizer/mul�
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOpC^wide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp2�
Bwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_2/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
n
(__inference_embedding_layer_call_fn_4279

inputs
unknown
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_embedding_layer_call_and_return_conditional_losses_29052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
=__inference_dnn_layer_call_and_return_conditional_losses_3149

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	1�*
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:����������2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������2
dense_1/Relu�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@2
dense_2/BiasAddp
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@2
dense_2/Relu~
dropout/IdentityIdentitydense_2/Relu:activations:0*
T0*'
_output_shapes
:���������@2
dropout/Identity�
IdentityIdentitydropout/Identity:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������1::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������1
 
_user_specified_nameinputs
�
�
E__inference_embedding_3_layer_call_and_return_conditional_losses_4356

inputs,
(embedding_lookup_readvariableop_resource
identity��embedding_lookup/ReadVariableOp�Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�
embedding_lookup/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype02!
embedding_lookup/ReadVariableOp�
embedding_lookup/axisConst*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*
_output_shapes
: *
dtype0*
value	B : 2
embedding_lookup/axis�
embedding_lookupGatherV2'embedding_lookup/ReadVariableOp:value:0inputsembedding_lookup/axis:output:0*
Taxis0*
Tindices0*
Tparams0*2
_class(
&$loc:@embedding_lookup/ReadVariableOp*'
_output_shapes
:���������2
embedding_lookup�
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*'
_output_shapes
:���������2
embedding_lookup/Identity�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(embedding_lookup_readvariableop_resource*
_output_shapes

:*
dtype02D
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp�
3wide_deep/embedding_3/embeddings/Regularizer/SquareSquareJwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:25
3wide_deep/embedding_3/embeddings/Regularizer/Square�
2wide_deep/embedding_3/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       24
2wide_deep/embedding_3/embeddings/Regularizer/Const�
0wide_deep/embedding_3/embeddings/Regularizer/SumSum7wide_deep/embedding_3/embeddings/Regularizer/Square:y:0;wide_deep/embedding_3/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/Sum�
2wide_deep/embedding_3/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��824
2wide_deep/embedding_3/embeddings/Regularizer/mul/x�
0wide_deep/embedding_3/embeddings/Regularizer/mulMul;wide_deep/embedding_3/embeddings/Regularizer/mul/x:output:09wide_deep/embedding_3/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 22
0wide_deep/embedding_3/embeddings/Regularizer/mul�
IdentityIdentity"embedding_lookup/Identity:output:0 ^embedding_lookup/ReadVariableOpC^wide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������:2B
embedding_lookup/ReadVariableOpembedding_lookup/ReadVariableOp2�
Bwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOpBwide_deep/embedding_3/embeddings/Regularizer/Square/ReadVariableOp:K G
#
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������	
;
input_20
serving_default_input_2:0���������<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�)
dense_feature_columns
sparse_feature_columns
embed_layers
dnn_network

linear
final_dense
	optimizer
regularization_losses
		variables

trainable_variables
	keras_api

signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"�&
_tf_keras_model�&{"class_name": "WideDeep", "name": "wide_deep", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "WideDeep"}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
_
0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
a
embed_0
embed_1
embed_2
embed_3
embed_4"
trackable_dict_wrapper
�
 dnn_network
!dropout
"regularization_losses
#	variables
$trainable_variables
%	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "DNN", "name": "dnn", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
�
	&dense
'regularization_losses
(	variables
)trainable_variables
*	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Linear", "name": "linear", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "linear", "trainable": true, "dtype": "float32"}}
�

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
�
1iter

2beta_1

3beta_2
	4decay
5learning_rate+m�,m�6m�7m�8m�9m�:m�;m�<m�=m�>m�?m�@m�Am�Bm�+v�,v�6v�7v�8v�9v�:v�;v�<v�=v�>v�?v�@v�Av�Bv�"
	optimizer
H
�0
�1
�2
�3
�4"
trackable_list_wrapper
�
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11
B12
+13
,14"
trackable_list_wrapper
�
60
71
82
93
:4
;5
<6
=7
>8
?9
@10
A11
B12
+13
,14"
trackable_list_wrapper
�
regularization_losses
Cnon_trainable_variables
		variables

Dlayers
Elayer_metrics

trainable_variables
Flayer_regularization_losses
Gmetrics
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
�
6
embeddings
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 39757, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
�
7
embeddings
Lregularization_losses
M	variables
Ntrainable_variables
O	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 18811, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
�
8
embeddings
Pregularization_losses
Q	variables
Rtrainable_variables
S	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_2", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 3000, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
�
9
embeddings
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 12, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
�
:
embeddings
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Embedding", "name": "embedding_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "input_dim": 76507, "output_dim": 8, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999747378752e-05}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [null]}}
5
\0
]1
^2"
trackable_list_wrapper
�
_regularization_losses
`	variables
atrainable_variables
b	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.0, "noise_shape": null, "seed": null}}
 "
trackable_list_wrapper
J
;0
<1
=2
>3
?4
@5"
trackable_list_wrapper
J
;0
<1
=2
>3
?4
@5"
trackable_list_wrapper
�
"regularization_losses
cnon_trainable_variables
#	variables
dlayer_metrics

elayers
$trainable_variables
flayer_regularization_losses
gmetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

Akernel
Bbias
hregularization_losses
i	variables
jtrainable_variables
k	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}}
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
�
'regularization_losses
lnon_trainable_variables
(	variables
mlayer_metrics

nlayers
)trainable_variables
olayer_regularization_losses
pmetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
*:(@2wide_deep/dense_4/kernel
$:"2wide_deep/dense_4/bias
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
�
-regularization_losses
qnon_trainable_variables
.	variables
rlayer_metrics

slayers
/trainable_variables
tlayer_regularization_losses
umetrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
2:0
Ͷ2wide_deep/embedding/embeddings
4:2
��2 wide_deep/embedding_1/embeddings
3:1	�2 wide_deep/embedding_2/embeddings
2:02 wide_deep/embedding_3/embeddings
4:2
��2 wide_deep/embedding_4/embeddings
-:+	1�2wide_deep/dnn/dense/kernel
':%�2wide_deep/dnn/dense/bias
0:.
��2wide_deep/dnn/dense_1/kernel
):'�2wide_deep/dnn/dense_1/bias
/:-	�@2wide_deep/dnn/dense_2/kernel
(:&@2wide_deep/dnn/dense_2/bias
1:/	2wide_deep/linear/dense_3/kernel
+:)2wide_deep/linear/dense_3/bias
 "
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
'
60"
trackable_list_wrapper
'
60"
trackable_list_wrapper
�
Hregularization_losses
xnon_trainable_variables
I	variables
ylayer_metrics

zlayers
Jtrainable_variables
{layer_regularization_losses
|metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
'
70"
trackable_list_wrapper
'
70"
trackable_list_wrapper
�
Lregularization_losses
}non_trainable_variables
M	variables
~layer_metrics

layers
Ntrainable_variables
 �layer_regularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
'
80"
trackable_list_wrapper
'
80"
trackable_list_wrapper
�
Pregularization_losses
�non_trainable_variables
Q	variables
�layer_metrics
�layers
Rtrainable_variables
 �layer_regularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
'
90"
trackable_list_wrapper
'
90"
trackable_list_wrapper
�
Tregularization_losses
�non_trainable_variables
U	variables
�layer_metrics
�layers
Vtrainable_variables
 �layer_regularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
(
�0"
trackable_list_wrapper
'
:0"
trackable_list_wrapper
'
:0"
trackable_list_wrapper
�
Xregularization_losses
�non_trainable_variables
Y	variables
�layer_metrics
�layers
Ztrainable_variables
 �layer_regularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�

;kernel
<bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
�

=kernel
>bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
�

?kernel
@bias
�regularization_losses
�	variables
�trainable_variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
_regularization_losses
�non_trainable_variables
`	variables
�layer_metrics
�layers
atrainable_variables
 �layer_regularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
<
\0
]1
^2
!3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
�
hregularization_losses
�non_trainable_variables
i	variables
�layer_metrics
�layers
jtrainable_variables
 �layer_regularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
&0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�"
�true_positives
�true_negatives
�false_positives
�false_negatives
�	variables
�	keras_api"�!
_tf_keras_metric�!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
�
�regularization_losses
�non_trainable_variables
�	variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
�
�regularization_losses
�non_trainable_variables
�	variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
.
?0
@1"
trackable_list_wrapper
�
�regularization_losses
�non_trainable_variables
�	variables
�layer_metrics
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:� (2true_positives
:� (2true_negatives
 :� (2false_positives
 :� (2false_negatives
@
�0
�1
�2
�3"
trackable_list_wrapper
.
�	variables"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
/:-@2Adam/wide_deep/dense_4/kernel/m
):'2Adam/wide_deep/dense_4/bias/m
7:5
Ͷ2%Adam/wide_deep/embedding/embeddings/m
9:7
��2'Adam/wide_deep/embedding_1/embeddings/m
8:6	�2'Adam/wide_deep/embedding_2/embeddings/m
7:52'Adam/wide_deep/embedding_3/embeddings/m
9:7
��2'Adam/wide_deep/embedding_4/embeddings/m
2:0	1�2!Adam/wide_deep/dnn/dense/kernel/m
,:*�2Adam/wide_deep/dnn/dense/bias/m
5:3
��2#Adam/wide_deep/dnn/dense_1/kernel/m
.:,�2!Adam/wide_deep/dnn/dense_1/bias/m
4:2	�@2#Adam/wide_deep/dnn/dense_2/kernel/m
-:+@2!Adam/wide_deep/dnn/dense_2/bias/m
6:4	2&Adam/wide_deep/linear/dense_3/kernel/m
0:.2$Adam/wide_deep/linear/dense_3/bias/m
/:-@2Adam/wide_deep/dense_4/kernel/v
):'2Adam/wide_deep/dense_4/bias/v
7:5
Ͷ2%Adam/wide_deep/embedding/embeddings/v
9:7
��2'Adam/wide_deep/embedding_1/embeddings/v
8:6	�2'Adam/wide_deep/embedding_2/embeddings/v
7:52'Adam/wide_deep/embedding_3/embeddings/v
9:7
��2'Adam/wide_deep/embedding_4/embeddings/v
2:0	1�2!Adam/wide_deep/dnn/dense/kernel/v
,:*�2Adam/wide_deep/dnn/dense/bias/v
5:3
��2#Adam/wide_deep/dnn/dense_1/kernel/v
.:,�2!Adam/wide_deep/dnn/dense_1/bias/v
4:2	�@2#Adam/wide_deep/dnn/dense_2/kernel/v
-:+@2!Adam/wide_deep/dnn/dense_2/bias/v
6:4	2&Adam/wide_deep/linear/dense_3/kernel/v
0:.2$Adam/wide_deep/linear/dense_3/bias/v
�2�
(__inference_wide_deep_layer_call_fn_3643
(__inference_wide_deep_layer_call_fn_3504
(__inference_wide_deep_layer_call_fn_4046
(__inference_wide_deep_layer_call_fn_4010�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
C__inference_wide_deep_layer_call_and_return_conditional_losses_3261
C__inference_wide_deep_layer_call_and_return_conditional_losses_3850
C__inference_wide_deep_layer_call_and_return_conditional_losses_3974
C__inference_wide_deep_layer_call_and_return_conditional_losses_3364�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
__inference__wrapped_model_2881�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *N�K
I�F
!�
input_1���������	
!�
input_2���������
�2�
"__inference_dnn_layer_call_fn_4122
"__inference_dnn_layer_call_fn_4139�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
=__inference_dnn_layer_call_and_return_conditional_losses_4105
=__inference_dnn_layer_call_and_return_conditional_losses_4079�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
%__inference_linear_layer_call_fn_4168
%__inference_linear_layer_call_fn_4177�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
@__inference_linear_layer_call_and_return_conditional_losses_4159
@__inference_linear_layer_call_and_return_conditional_losses_4149�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkwjkwargs
defaults� 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�2�
&__inference_dense_4_layer_call_fn_4196�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_dense_4_layer_call_and_return_conditional_losses_4187�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_4207�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_4218�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_4229�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_4240�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_4_4251�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
"__inference_signature_wrapper_3719input_1input_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_embedding_layer_call_fn_4279�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_embedding_layer_call_and_return_conditional_losses_4272�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_embedding_1_layer_call_fn_4307�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_embedding_1_layer_call_and_return_conditional_losses_4300�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_embedding_2_layer_call_fn_4335�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_embedding_2_layer_call_and_return_conditional_losses_4328�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_embedding_3_layer_call_fn_4363�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_embedding_3_layer_call_and_return_conditional_losses_4356�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_embedding_4_layer_call_fn_4391�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_embedding_4_layer_call_and_return_conditional_losses_4384�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
__inference__wrapped_model_2881�6789:AB;<=>?@+,X�U
N�K
I�F
!�
input_1���������	
!�
input_2���������
� "3�0
.
output_1"�
output_1����������
A__inference_dense_4_layer_call_and_return_conditional_losses_4187\+,/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� y
&__inference_dense_4_layer_call_fn_4196O+,/�,
%�"
 �
inputs���������@
� "�����������
=__inference_dnn_layer_call_and_return_conditional_losses_4079p;<=>?@?�<
%�"
 �
inputs���������1
�

trainingp"%�"
�
0���������@
� �
=__inference_dnn_layer_call_and_return_conditional_losses_4105p;<=>?@?�<
%�"
 �
inputs���������1
�

trainingp "%�"
�
0���������@
� �
"__inference_dnn_layer_call_fn_4122c;<=>?@?�<
%�"
 �
inputs���������1
�

trainingp"����������@�
"__inference_dnn_layer_call_fn_4139c;<=>?@?�<
%�"
 �
inputs���������1
�

trainingp "����������@�
E__inference_embedding_1_layer_call_and_return_conditional_losses_4300W7+�(
!�
�
inputs���������
� "%�"
�
0���������
� x
*__inference_embedding_1_layer_call_fn_4307J7+�(
!�
�
inputs���������
� "�����������
E__inference_embedding_2_layer_call_and_return_conditional_losses_4328W8+�(
!�
�
inputs���������
� "%�"
�
0���������
� x
*__inference_embedding_2_layer_call_fn_4335J8+�(
!�
�
inputs���������
� "�����������
E__inference_embedding_3_layer_call_and_return_conditional_losses_4356W9+�(
!�
�
inputs���������
� "%�"
�
0���������
� x
*__inference_embedding_3_layer_call_fn_4363J9+�(
!�
�
inputs���������
� "�����������
E__inference_embedding_4_layer_call_and_return_conditional_losses_4384W:+�(
!�
�
inputs���������
� "%�"
�
0���������
� x
*__inference_embedding_4_layer_call_fn_4391J:+�(
!�
�
inputs���������
� "�����������
C__inference_embedding_layer_call_and_return_conditional_losses_4272W6+�(
!�
�
inputs���������
� "%�"
�
0���������
� v
(__inference_embedding_layer_call_fn_4279J6+�(
!�
�
inputs���������
� "�����������
@__inference_linear_layer_call_and_return_conditional_losses_4149lAB?�<
%�"
 �
inputs���������	
�

trainingp"%�"
�
0���������
� �
@__inference_linear_layer_call_and_return_conditional_losses_4159lAB?�<
%�"
 �
inputs���������	
�

trainingp "%�"
�
0���������
� �
%__inference_linear_layer_call_fn_4168_AB?�<
%�"
 �
inputs���������	
�

trainingp"�����������
%__inference_linear_layer_call_fn_4177_AB?�<
%�"
 �
inputs���������	
�

trainingp "����������9
__inference_loss_fn_0_42076�

� 
� "� 9
__inference_loss_fn_1_42187�

� 
� "� 9
__inference_loss_fn_2_42298�

� 
� "� 9
__inference_loss_fn_3_42409�

� 
� "� 9
__inference_loss_fn_4_4251:�

� 
� "� �
"__inference_signature_wrapper_3719�6789:AB;<=>?@+,i�f
� 
_�\
,
input_1!�
input_1���������	
,
input_2!�
input_2���������"3�0
.
output_1"�
output_1����������
C__inference_wide_deep_layer_call_and_return_conditional_losses_3261�6789:AB;<=>?@+,h�e
N�K
I�F
!�
input_1���������	
!�
input_2���������
�

trainingp"%�"
�
0���������
� �
C__inference_wide_deep_layer_call_and_return_conditional_losses_3364�6789:AB;<=>?@+,h�e
N�K
I�F
!�
input_1���������	
!�
input_2���������
�

trainingp "%�"
�
0���������
� �
C__inference_wide_deep_layer_call_and_return_conditional_losses_3850�6789:AB;<=>?@+,j�g
P�M
K�H
"�
inputs/0���������	
"�
inputs/1���������
�

trainingp"%�"
�
0���������
� �
C__inference_wide_deep_layer_call_and_return_conditional_losses_3974�6789:AB;<=>?@+,j�g
P�M
K�H
"�
inputs/0���������	
"�
inputs/1���������
�

trainingp "%�"
�
0���������
� �
(__inference_wide_deep_layer_call_fn_3504�6789:AB;<=>?@+,h�e
N�K
I�F
!�
input_1���������	
!�
input_2���������
�

trainingp"�����������
(__inference_wide_deep_layer_call_fn_3643�6789:AB;<=>?@+,h�e
N�K
I�F
!�
input_1���������	
!�
input_2���������
�

trainingp "�����������
(__inference_wide_deep_layer_call_fn_4010�6789:AB;<=>?@+,j�g
P�M
K�H
"�
inputs/0���������	
"�
inputs/1���������
�

trainingp"�����������
(__inference_wide_deep_layer_call_fn_4046�6789:AB;<=>?@+,j�g
P�M
K�H
"�
inputs/0���������	
"�
inputs/1���������
�

trainingp "����������