б╔"
Я░
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

ч
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%иЛ8"&
exponential_avg_factorfloat%  ђ?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
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
ѓ
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
Ї
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
є
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ѕ
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
dtypetypeѕ
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
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
┴
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
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ЊЕ
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
Є
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*&
shared_nameAdam/dense_1/kernel/v
ђ
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	ђ*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:ђ*
dtype0
Ѓ
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ђ*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	@ђ*
dtype0
к
7Adam/convolutional_block_3/batch_normalization_3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/convolutional_block_3/batch_normalization_3/beta/v
┐
KAdam/convolutional_block_3/batch_normalization_3/beta/v/Read/ReadVariableOpReadVariableOp7Adam/convolutional_block_3/batch_normalization_3/beta/v*
_output_shapes
:@*
dtype0
╚
8Adam/convolutional_block_3/batch_normalization_3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8Adam/convolutional_block_3/batch_normalization_3/gamma/v
┴
LAdam/convolutional_block_3/batch_normalization_3/gamma/v/Read/ReadVariableOpReadVariableOp8Adam/convolutional_block_3/batch_normalization_3/gamma/v*
_output_shapes
:@*
dtype0
г
*Adam/convolutional_block_3/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/convolutional_block_3/conv2d_3/bias/v
Ц
>Adam/convolutional_block_3/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOp*Adam/convolutional_block_3/conv2d_3/bias/v*
_output_shapes
:@*
dtype0
╝
,Adam/convolutional_block_3/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*=
shared_name.,Adam/convolutional_block_3/conv2d_3/kernel/v
х
@Adam/convolutional_block_3/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/convolutional_block_3/conv2d_3/kernel/v*&
_output_shapes
:@@*
dtype0
к
7Adam/convolutional_block_2/batch_normalization_2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/convolutional_block_2/batch_normalization_2/beta/v
┐
KAdam/convolutional_block_2/batch_normalization_2/beta/v/Read/ReadVariableOpReadVariableOp7Adam/convolutional_block_2/batch_normalization_2/beta/v*
_output_shapes
:@*
dtype0
╚
8Adam/convolutional_block_2/batch_normalization_2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8Adam/convolutional_block_2/batch_normalization_2/gamma/v
┴
LAdam/convolutional_block_2/batch_normalization_2/gamma/v/Read/ReadVariableOpReadVariableOp8Adam/convolutional_block_2/batch_normalization_2/gamma/v*
_output_shapes
:@*
dtype0
г
*Adam/convolutional_block_2/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/convolutional_block_2/conv2d_2/bias/v
Ц
>Adam/convolutional_block_2/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOp*Adam/convolutional_block_2/conv2d_2/bias/v*
_output_shapes
:@*
dtype0
╝
,Adam/convolutional_block_2/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*=
shared_name.,Adam/convolutional_block_2/conv2d_2/kernel/v
х
@Adam/convolutional_block_2/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/convolutional_block_2/conv2d_2/kernel/v*&
_output_shapes
: @*
dtype0
к
7Adam/convolutional_block_1/batch_normalization_1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/convolutional_block_1/batch_normalization_1/beta/v
┐
KAdam/convolutional_block_1/batch_normalization_1/beta/v/Read/ReadVariableOpReadVariableOp7Adam/convolutional_block_1/batch_normalization_1/beta/v*
_output_shapes
: *
dtype0
╚
8Adam/convolutional_block_1/batch_normalization_1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/convolutional_block_1/batch_normalization_1/gamma/v
┴
LAdam/convolutional_block_1/batch_normalization_1/gamma/v/Read/ReadVariableOpReadVariableOp8Adam/convolutional_block_1/batch_normalization_1/gamma/v*
_output_shapes
: *
dtype0
г
*Adam/convolutional_block_1/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/convolutional_block_1/conv2d_1/bias/v
Ц
>Adam/convolutional_block_1/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOp*Adam/convolutional_block_1/conv2d_1/bias/v*
_output_shapes
: *
dtype0
╝
,Adam/convolutional_block_1/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *=
shared_name.,Adam/convolutional_block_1/conv2d_1/kernel/v
х
@Adam/convolutional_block_1/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/convolutional_block_1/conv2d_1/kernel/v*&
_output_shapes
:  *
dtype0
Й
3Adam/convolutional_block/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/convolutional_block/batch_normalization/beta/v
и
GAdam/convolutional_block/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOp3Adam/convolutional_block/batch_normalization/beta/v*
_output_shapes
: *
dtype0
└
4Adam/convolutional_block/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *E
shared_name64Adam/convolutional_block/batch_normalization/gamma/v
╣
HAdam/convolutional_block/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp4Adam/convolutional_block/batch_normalization/gamma/v*
_output_shapes
: *
dtype0
ц
&Adam/convolutional_block/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/convolutional_block/conv2d/bias/v
Ю
:Adam/convolutional_block/conv2d/bias/v/Read/ReadVariableOpReadVariableOp&Adam/convolutional_block/conv2d/bias/v*
_output_shapes
: *
dtype0
┤
(Adam/convolutional_block/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/convolutional_block/conv2d/kernel/v
Г
<Adam/convolutional_block/conv2d/kernel/v/Read/ReadVariableOpReadVariableOp(Adam/convolutional_block/conv2d/kernel/v*&
_output_shapes
: *
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
Є
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*&
shared_nameAdam/dense_1/kernel/m
ђ
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	ђ*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:ђ*
dtype0
Ѓ
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ђ*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	@ђ*
dtype0
к
7Adam/convolutional_block_3/batch_normalization_3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/convolutional_block_3/batch_normalization_3/beta/m
┐
KAdam/convolutional_block_3/batch_normalization_3/beta/m/Read/ReadVariableOpReadVariableOp7Adam/convolutional_block_3/batch_normalization_3/beta/m*
_output_shapes
:@*
dtype0
╚
8Adam/convolutional_block_3/batch_normalization_3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8Adam/convolutional_block_3/batch_normalization_3/gamma/m
┴
LAdam/convolutional_block_3/batch_normalization_3/gamma/m/Read/ReadVariableOpReadVariableOp8Adam/convolutional_block_3/batch_normalization_3/gamma/m*
_output_shapes
:@*
dtype0
г
*Adam/convolutional_block_3/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/convolutional_block_3/conv2d_3/bias/m
Ц
>Adam/convolutional_block_3/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOp*Adam/convolutional_block_3/conv2d_3/bias/m*
_output_shapes
:@*
dtype0
╝
,Adam/convolutional_block_3/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*=
shared_name.,Adam/convolutional_block_3/conv2d_3/kernel/m
х
@Adam/convolutional_block_3/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/convolutional_block_3/conv2d_3/kernel/m*&
_output_shapes
:@@*
dtype0
к
7Adam/convolutional_block_2/batch_normalization_2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97Adam/convolutional_block_2/batch_normalization_2/beta/m
┐
KAdam/convolutional_block_2/batch_normalization_2/beta/m/Read/ReadVariableOpReadVariableOp7Adam/convolutional_block_2/batch_normalization_2/beta/m*
_output_shapes
:@*
dtype0
╚
8Adam/convolutional_block_2/batch_normalization_2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8Adam/convolutional_block_2/batch_normalization_2/gamma/m
┴
LAdam/convolutional_block_2/batch_normalization_2/gamma/m/Read/ReadVariableOpReadVariableOp8Adam/convolutional_block_2/batch_normalization_2/gamma/m*
_output_shapes
:@*
dtype0
г
*Adam/convolutional_block_2/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*;
shared_name,*Adam/convolutional_block_2/conv2d_2/bias/m
Ц
>Adam/convolutional_block_2/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOp*Adam/convolutional_block_2/conv2d_2/bias/m*
_output_shapes
:@*
dtype0
╝
,Adam/convolutional_block_2/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*=
shared_name.,Adam/convolutional_block_2/conv2d_2/kernel/m
х
@Adam/convolutional_block_2/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/convolutional_block_2/conv2d_2/kernel/m*&
_output_shapes
: @*
dtype0
к
7Adam/convolutional_block_1/batch_normalization_1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/convolutional_block_1/batch_normalization_1/beta/m
┐
KAdam/convolutional_block_1/batch_normalization_1/beta/m/Read/ReadVariableOpReadVariableOp7Adam/convolutional_block_1/batch_normalization_1/beta/m*
_output_shapes
: *
dtype0
╚
8Adam/convolutional_block_1/batch_normalization_1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *I
shared_name:8Adam/convolutional_block_1/batch_normalization_1/gamma/m
┴
LAdam/convolutional_block_1/batch_normalization_1/gamma/m/Read/ReadVariableOpReadVariableOp8Adam/convolutional_block_1/batch_normalization_1/gamma/m*
_output_shapes
: *
dtype0
г
*Adam/convolutional_block_1/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *;
shared_name,*Adam/convolutional_block_1/conv2d_1/bias/m
Ц
>Adam/convolutional_block_1/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOp*Adam/convolutional_block_1/conv2d_1/bias/m*
_output_shapes
: *
dtype0
╝
,Adam/convolutional_block_1/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *=
shared_name.,Adam/convolutional_block_1/conv2d_1/kernel/m
х
@Adam/convolutional_block_1/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/convolutional_block_1/conv2d_1/kernel/m*&
_output_shapes
:  *
dtype0
Й
3Adam/convolutional_block/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53Adam/convolutional_block/batch_normalization/beta/m
и
GAdam/convolutional_block/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOp3Adam/convolutional_block/batch_normalization/beta/m*
_output_shapes
: *
dtype0
└
4Adam/convolutional_block/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *E
shared_name64Adam/convolutional_block/batch_normalization/gamma/m
╣
HAdam/convolutional_block/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp4Adam/convolutional_block/batch_normalization/gamma/m*
_output_shapes
: *
dtype0
ц
&Adam/convolutional_block/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&Adam/convolutional_block/conv2d/bias/m
Ю
:Adam/convolutional_block/conv2d/bias/m/Read/ReadVariableOpReadVariableOp&Adam/convolutional_block/conv2d/bias/m*
_output_shapes
: *
dtype0
┤
(Adam/convolutional_block/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *9
shared_name*(Adam/convolutional_block/conv2d/kernel/m
Г
<Adam/convolutional_block/conv2d/kernel/m/Read/ReadVariableOpReadVariableOp(Adam/convolutional_block/conv2d/kernel/m*&
_output_shapes
: *
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
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
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
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ђ*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	ђ*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ђ*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:ђ*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	@ђ*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	@ђ*
dtype0
╬
;convolutional_block_3/batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*L
shared_name=;convolutional_block_3/batch_normalization_3/moving_variance
К
Oconvolutional_block_3/batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp;convolutional_block_3/batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0
к
7convolutional_block_3/batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97convolutional_block_3/batch_normalization_3/moving_mean
┐
Kconvolutional_block_3/batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp7convolutional_block_3/batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
И
0convolutional_block_3/batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20convolutional_block_3/batch_normalization_3/beta
▒
Dconvolutional_block_3/batch_normalization_3/beta/Read/ReadVariableOpReadVariableOp0convolutional_block_3/batch_normalization_3/beta*
_output_shapes
:@*
dtype0
║
1convolutional_block_3/batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31convolutional_block_3/batch_normalization_3/gamma
│
Econvolutional_block_3/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOp1convolutional_block_3/batch_normalization_3/gamma*
_output_shapes
:@*
dtype0
ъ
#convolutional_block_3/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#convolutional_block_3/conv2d_3/bias
Ќ
7convolutional_block_3/conv2d_3/bias/Read/ReadVariableOpReadVariableOp#convolutional_block_3/conv2d_3/bias*
_output_shapes
:@*
dtype0
«
%convolutional_block_3/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*6
shared_name'%convolutional_block_3/conv2d_3/kernel
Д
9convolutional_block_3/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp%convolutional_block_3/conv2d_3/kernel*&
_output_shapes
:@@*
dtype0
╬
;convolutional_block_2/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*L
shared_name=;convolutional_block_2/batch_normalization_2/moving_variance
К
Oconvolutional_block_2/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp;convolutional_block_2/batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
к
7convolutional_block_2/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*H
shared_name97convolutional_block_2/batch_normalization_2/moving_mean
┐
Kconvolutional_block_2/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp7convolutional_block_2/batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
И
0convolutional_block_2/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*A
shared_name20convolutional_block_2/batch_normalization_2/beta
▒
Dconvolutional_block_2/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp0convolutional_block_2/batch_normalization_2/beta*
_output_shapes
:@*
dtype0
║
1convolutional_block_2/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*B
shared_name31convolutional_block_2/batch_normalization_2/gamma
│
Econvolutional_block_2/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp1convolutional_block_2/batch_normalization_2/gamma*
_output_shapes
:@*
dtype0
ъ
#convolutional_block_2/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#convolutional_block_2/conv2d_2/bias
Ќ
7convolutional_block_2/conv2d_2/bias/Read/ReadVariableOpReadVariableOp#convolutional_block_2/conv2d_2/bias*
_output_shapes
:@*
dtype0
«
%convolutional_block_2/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%convolutional_block_2/conv2d_2/kernel
Д
9convolutional_block_2/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp%convolutional_block_2/conv2d_2/kernel*&
_output_shapes
: @*
dtype0
╬
;convolutional_block_1/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *L
shared_name=;convolutional_block_1/batch_normalization_1/moving_variance
К
Oconvolutional_block_1/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp;convolutional_block_1/batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
к
7convolutional_block_1/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97convolutional_block_1/batch_normalization_1/moving_mean
┐
Kconvolutional_block_1/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp7convolutional_block_1/batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
И
0convolutional_block_1/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20convolutional_block_1/batch_normalization_1/beta
▒
Dconvolutional_block_1/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp0convolutional_block_1/batch_normalization_1/beta*
_output_shapes
: *
dtype0
║
1convolutional_block_1/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *B
shared_name31convolutional_block_1/batch_normalization_1/gamma
│
Econvolutional_block_1/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp1convolutional_block_1/batch_normalization_1/gamma*
_output_shapes
: *
dtype0
ъ
#convolutional_block_1/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#convolutional_block_1/conv2d_1/bias
Ќ
7convolutional_block_1/conv2d_1/bias/Read/ReadVariableOpReadVariableOp#convolutional_block_1/conv2d_1/bias*
_output_shapes
: *
dtype0
«
%convolutional_block_1/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *6
shared_name'%convolutional_block_1/conv2d_1/kernel
Д
9convolutional_block_1/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp%convolutional_block_1/conv2d_1/kernel*&
_output_shapes
:  *
dtype0
к
7convolutional_block/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97convolutional_block/batch_normalization/moving_variance
┐
Kconvolutional_block/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp7convolutional_block/batch_normalization/moving_variance*
_output_shapes
: *
dtype0
Й
3convolutional_block/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53convolutional_block/batch_normalization/moving_mean
и
Gconvolutional_block/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp3convolutional_block/batch_normalization/moving_mean*
_output_shapes
: *
dtype0
░
,convolutional_block/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *=
shared_name.,convolutional_block/batch_normalization/beta
Е
@convolutional_block/batch_normalization/beta/Read/ReadVariableOpReadVariableOp,convolutional_block/batch_normalization/beta*
_output_shapes
: *
dtype0
▓
-convolutional_block/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-convolutional_block/batch_normalization/gamma
Ф
Aconvolutional_block/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp-convolutional_block/batch_normalization/gamma*
_output_shapes
: *
dtype0
ќ
convolutional_block/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!convolutional_block/conv2d/bias
Ј
3convolutional_block/conv2d/bias/Read/ReadVariableOpReadVariableOpconvolutional_block/conv2d/bias*
_output_shapes
: *
dtype0
д
!convolutional_block/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!convolutional_block/conv2d/kernel
Ъ
5convolutional_block/conv2d/kernel/Read/ReadVariableOpReadVariableOp!convolutional_block/conv2d/kernel*&
_output_shapes
: *
dtype0
ј
serving_default_input_1Placeholder*1
_output_shapes
:         ђђ*
dtype0*&
shape:         ђђ
е
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1!convolutional_block/conv2d/kernelconvolutional_block/conv2d/bias-convolutional_block/batch_normalization/gamma,convolutional_block/batch_normalization/beta3convolutional_block/batch_normalization/moving_mean7convolutional_block/batch_normalization/moving_variance%convolutional_block_1/conv2d_1/kernel#convolutional_block_1/conv2d_1/bias1convolutional_block_1/batch_normalization_1/gamma0convolutional_block_1/batch_normalization_1/beta7convolutional_block_1/batch_normalization_1/moving_mean;convolutional_block_1/batch_normalization_1/moving_variance%convolutional_block_2/conv2d_2/kernel#convolutional_block_2/conv2d_2/bias1convolutional_block_2/batch_normalization_2/gamma0convolutional_block_2/batch_normalization_2/beta7convolutional_block_2/batch_normalization_2/moving_mean;convolutional_block_2/batch_normalization_2/moving_variance%convolutional_block_3/conv2d_3/kernel#convolutional_block_3/conv2d_3/bias1convolutional_block_3/batch_normalization_3/gamma0convolutional_block_3/batch_normalization_3/beta7convolutional_block_3/batch_normalization_3/moving_mean;convolutional_block_3/batch_normalization_3/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/bias*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *,
f'R%
#__inference_signature_wrapper_66785

NoOpNoOp
Э╚
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*▓╚
valueД╚BБ╚ BЏ╚
Ј
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
convolutional_portion
	average_pooling

dense_portion
	optimizer

signatures*
┌
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21
#22
$23
%24
&25
'26
(27*
џ
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
%16
&17
'18
(19*
* 
░
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
.trace_0
/trace_1
0trace_2
1trace_3* 
6
2trace_0
3trace_1
4trace_2
5trace_3* 
* 
г
6layer_with_weights-0
6layer-0
7layer_with_weights-1
7layer-1
8layer_with_weights-2
8layer-2
9layer_with_weights-3
9layer-3
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses*
ј
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses* 
в
Flayer_with_weights-0
Flayer-0
Glayer-1
Hlayer_with_weights-1
Hlayer-2
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses*
н
Oiter

Pbeta_1

Qbeta_2
	Rdecay
Slearning_ratemБmцmЦmдmДmеmЕmфmФmгmГm«m» m░!m▒"m▓%m│&m┤'mх(mХvиvИv╣v║v╗v╝vйvЙv┐v└v┴v┬v├ v─!v┼"vк%vК&v╚'v╔(v╩*

Tserving_default* 
a[
VARIABLE_VALUE!convolutional_block/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEconvolutional_block/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE-convolutional_block/batch_normalization/gamma&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
lf
VARIABLE_VALUE,convolutional_block/batch_normalization/beta&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUE3convolutional_block/batch_normalization/moving_mean&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE7convolutional_block/batch_normalization/moving_variance&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%convolutional_block_1/conv2d_1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#convolutional_block_1/conv2d_1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE1convolutional_block_1/batch_normalization_1/gamma&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE0convolutional_block_1/batch_normalization_1/beta&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7convolutional_block_1/batch_normalization_1/moving_mean'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;convolutional_block_1/batch_normalization_1/moving_variance'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%convolutional_block_2/conv2d_2/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#convolutional_block_2/conv2d_2/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1convolutional_block_2/batch_normalization_2/gamma'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0convolutional_block_2/batch_normalization_2/beta'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7convolutional_block_2/batch_normalization_2/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;convolutional_block_2/batch_normalization_2/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUE%convolutional_block_3/conv2d_3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#convolutional_block_3/conv2d_3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUE1convolutional_block_3/batch_normalization_3/gamma'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE0convolutional_block_3/batch_normalization_3/beta'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE7convolutional_block_3/batch_normalization_3/moving_mean'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE;convolutional_block_3/batch_normalization_3/moving_variance'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
dense/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEdense_1/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEdense_1/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
<
0
1
2
3
4
5
#6
$7*

0
	1

2*

U0
V1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
К
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]conv_2d
^max_pool_2d
_batch_normalization*
К
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses
fconv_2d
gmax_pool_2d
hbatch_normalization*
К
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
oconv_2d
pmax_pool_2d
qbatch_normalization*
К
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
xconv_2d
ymax_pool_2d
zbatch_normalization*
║
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21
#22
$23*
z
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15*
* 
Њ
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses*
:
ђtrace_0
Ђtrace_1
ѓtrace_2
Ѓtrace_3* 
:
ёtrace_0
Ёtrace_1
єtrace_2
Єtrace_3* 
* 
* 
* 
ќ
ѕnon_trainable_variables
Ѕlayers
іmetrics
 Іlayer_regularization_losses
їlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses* 

Їtrace_0* 

јtrace_0* 
г
Ј	variables
љtrainable_variables
Љregularization_losses
њ	keras_api
Њ__call__
+ћ&call_and_return_all_conditional_losses

%kernel
&bias*
г
Ћ	variables
ќtrainable_variables
Ќregularization_losses
ў	keras_api
Ў__call__
+џ&call_and_return_all_conditional_losses
Џ_random_generator* 
г
ю	variables
Юtrainable_variables
ъregularization_losses
Ъ	keras_api
а__call__
+А&call_and_return_all_conditional_losses

'kernel
(bias*
 
%0
&1
'2
(3*
 
%0
&1
'2
(3*
* 
ў
бnon_trainable_variables
Бlayers
цmetrics
 Цlayer_regularization_losses
дlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*
:
Дtrace_0
еtrace_1
Еtrace_2
фtrace_3* 
:
Фtrace_0
гtrace_1
Гtrace_2
«trace_3* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
<
»	variables
░	keras_api

▒total

▓count*
M
│	variables
┤	keras_api

хtotal

Хcount
и
_fn_kwargs*
.
0
1
2
3
4
5*
 
0
1
2
3*
* 
ў
Иnon_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*
:
йtrace_0
Йtrace_1
┐trace_2
└trace_3* 
:
┴trace_0
┬trace_1
├trace_2
─trace_3* 
¤
┼	variables
кtrainable_variables
Кregularization_losses
╚	keras_api
╔__call__
+╩&call_and_return_all_conditional_losses

kernel
bias
!╦_jit_compiled_convolution_op*
ћ
╠	variables
═trainable_variables
╬regularization_losses
¤	keras_api
л__call__
+Л&call_and_return_all_conditional_losses* 
▄
м	variables
Мtrainable_variables
нregularization_losses
Н	keras_api
о__call__
+О&call_and_return_all_conditional_losses
	пaxis
	gamma
beta
moving_mean
moving_variance*
.
0
1
2
3
4
5*
 
0
1
2
3*
* 
ў
┘non_trainable_variables
┌layers
█metrics
 ▄layer_regularization_losses
Пlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses*

яtrace_0
▀trace_1* 

Яtrace_0
рtrace_1* 
¤
Р	variables
сtrainable_variables
Сregularization_losses
т	keras_api
Т__call__
+у&call_and_return_all_conditional_losses

kernel
bias
!У_jit_compiled_convolution_op*
ћ
ж	variables
Жtrainable_variables
вregularization_losses
В	keras_api
ь__call__
+Ь&call_and_return_all_conditional_losses* 
▄
№	variables
­trainable_variables
ыregularization_losses
Ы	keras_api
з__call__
+З&call_and_return_all_conditional_losses
	шaxis
	gamma
beta
moving_mean
moving_variance*
.
0
1
2
3
4
5*
 
0
1
2
3*
* 
ў
Шnon_trainable_variables
эlayers
Эmetrics
 щlayer_regularization_losses
Щlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses*

чtrace_0
Чtrace_1* 

§trace_0
■trace_1* 
¤
 	variables
ђtrainable_variables
Ђregularization_losses
ѓ	keras_api
Ѓ__call__
+ё&call_and_return_all_conditional_losses

kernel
bias
!Ё_jit_compiled_convolution_op*
ћ
є	variables
Єtrainable_variables
ѕregularization_losses
Ѕ	keras_api
і__call__
+І&call_and_return_all_conditional_losses* 
▄
ї	variables
Їtrainable_variables
јregularization_losses
Ј	keras_api
љ__call__
+Љ&call_and_return_all_conditional_losses
	њaxis
	gamma
beta
moving_mean
moving_variance*
.
0
 1
!2
"3
#4
$5*
 
0
 1
!2
"3*
* 
ў
Њnon_trainable_variables
ћlayers
Ћmetrics
 ќlayer_regularization_losses
Ќlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses*

ўtrace_0
Ўtrace_1* 

џtrace_0
Џtrace_1* 
¤
ю	variables
Юtrainable_variables
ъregularization_losses
Ъ	keras_api
а__call__
+А&call_and_return_all_conditional_losses

kernel
 bias
!б_jit_compiled_convolution_op*
ћ
Б	variables
цtrainable_variables
Цregularization_losses
д	keras_api
Д__call__
+е&call_and_return_all_conditional_losses* 
▄
Е	variables
фtrainable_variables
Фregularization_losses
г	keras_api
Г__call__
+«&call_and_return_all_conditional_losses
	»axis
	!gamma
"beta
#moving_mean
$moving_variance*
<
0
1
2
3
4
5
#6
$7*
 
60
71
82
93*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

%0
&1*

%0
&1*
* 
ъ
░non_trainable_variables
▒layers
▓metrics
 │layer_regularization_losses
┤layer_metrics
Ј	variables
љtrainable_variables
Љregularization_losses
Њ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses*

хtrace_0* 

Хtrace_0* 
* 
* 
* 
ю
иnon_trainable_variables
Иlayers
╣metrics
 ║layer_regularization_losses
╗layer_metrics
Ћ	variables
ќtrainable_variables
Ќregularization_losses
Ў__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses* 

╝trace_0
йtrace_1* 

Йtrace_0
┐trace_1* 
* 

'0
(1*

'0
(1*
* 
ъ
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
ю	variables
Юtrainable_variables
ъregularization_losses
а__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses*

┼trace_0* 

кtrace_0* 
* 

F0
G1
H2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

▒0
▓1*

»	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

х0
Х1*

│	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

]0
^1
_2*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
ъ
Кnon_trainable_variables
╚layers
╔metrics
 ╩layer_regularization_losses
╦layer_metrics
┼	variables
кtrainable_variables
Кregularization_losses
╔__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses*

╠trace_0* 

═trace_0* 
* 
* 
* 
* 
ю
╬non_trainable_variables
¤layers
лmetrics
 Лlayer_regularization_losses
мlayer_metrics
╠	variables
═trainable_variables
╬regularization_losses
л__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses* 

Мtrace_0* 

нtrace_0* 
 
0
1
2
3*

0
1*
* 
ъ
Нnon_trainable_variables
оlayers
Оmetrics
 пlayer_regularization_losses
┘layer_metrics
м	variables
Мtrainable_variables
нregularization_losses
о__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses*

┌trace_0
█trace_1* 

▄trace_0
Пtrace_1* 
* 

0
1*

f0
g1
h2*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
ъ
яnon_trainable_variables
▀layers
Яmetrics
 рlayer_regularization_losses
Рlayer_metrics
Р	variables
сtrainable_variables
Сregularization_losses
Т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses*

сtrace_0* 

Сtrace_0* 
* 
* 
* 
* 
ю
тnon_trainable_variables
Тlayers
уmetrics
 Уlayer_regularization_losses
жlayer_metrics
ж	variables
Жtrainable_variables
вregularization_losses
ь__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses* 

Жtrace_0* 

вtrace_0* 
 
0
1
2
3*

0
1*
* 
ъ
Вnon_trainable_variables
ьlayers
Ьmetrics
 №layer_regularization_losses
­layer_metrics
№	variables
­trainable_variables
ыregularization_losses
з__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses*

ыtrace_0
Ыtrace_1* 

зtrace_0
Зtrace_1* 
* 

0
1*

o0
p1
q2*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
ъ
шnon_trainable_variables
Шlayers
эmetrics
 Эlayer_regularization_losses
щlayer_metrics
 	variables
ђtrainable_variables
Ђregularization_losses
Ѓ__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses*

Щtrace_0* 

чtrace_0* 
* 
* 
* 
* 
ю
Чnon_trainable_variables
§layers
■metrics
  layer_regularization_losses
ђlayer_metrics
є	variables
Єtrainable_variables
ѕregularization_losses
і__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses* 

Ђtrace_0* 

ѓtrace_0* 
 
0
1
2
3*

0
1*
* 
ъ
Ѓnon_trainable_variables
ёlayers
Ёmetrics
 єlayer_regularization_losses
Єlayer_metrics
ї	variables
Їtrainable_variables
јregularization_losses
љ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses*

ѕtrace_0
Ѕtrace_1* 

іtrace_0
Іtrace_1* 
* 

#0
$1*

x0
y1
z2*
* 
* 
* 
* 
* 
* 
* 

0
 1*

0
 1*
* 
ъ
їnon_trainable_variables
Їlayers
јmetrics
 Јlayer_regularization_losses
љlayer_metrics
ю	variables
Юtrainable_variables
ъregularization_losses
а__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses*

Љtrace_0* 

њtrace_0* 
* 
* 
* 
* 
ю
Њnon_trainable_variables
ћlayers
Ћmetrics
 ќlayer_regularization_losses
Ќlayer_metrics
Б	variables
цtrainable_variables
Цregularization_losses
Д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses* 

ўtrace_0* 

Ўtrace_0* 
 
!0
"1
#2
$3*

!0
"1*
* 
ъ
џnon_trainable_variables
Џlayers
юmetrics
 Юlayer_regularization_losses
ъlayer_metrics
Е	variables
фtrainable_variables
Фregularization_losses
Г__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses*

Ъtrace_0
аtrace_1* 

Аtrace_0
бtrace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

#0
$1*
* 
* 
* 
* 
* 
* 
* 
* 
ё~
VARIABLE_VALUE(Adam/convolutional_block/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUE&Adam/convolutional_block/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Љі
VARIABLE_VALUE4Adam/convolutional_block/batch_normalization/gamma/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
љЅ
VARIABLE_VALUE3Adam/convolutional_block/batch_normalization/beta/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ѕѓ
VARIABLE_VALUE,Adam/convolutional_block_1/conv2d_1/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Єђ
VARIABLE_VALUE*Adam/convolutional_block_1/conv2d_1/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ћј
VARIABLE_VALUE8Adam/convolutional_block_1/batch_normalization_1/gamma/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ћЇ
VARIABLE_VALUE7Adam/convolutional_block_1/batch_normalization_1/beta/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
іЃ
VARIABLE_VALUE,Adam/convolutional_block_2/conv2d_2/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѕЂ
VARIABLE_VALUE*Adam/convolutional_block_2/conv2d_2/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ќЈ
VARIABLE_VALUE8Adam/convolutional_block_2/batch_normalization_2/gamma/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ћј
VARIABLE_VALUE7Adam/convolutional_block_2/batch_normalization_2/beta/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
іЃ
VARIABLE_VALUE,Adam/convolutional_block_3/conv2d_3/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѕЂ
VARIABLE_VALUE*Adam/convolutional_block_3/conv2d_3/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ќЈ
VARIABLE_VALUE8Adam/convolutional_block_3/batch_normalization_3/gamma/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ћј
VARIABLE_VALUE7Adam/convolutional_block_3/batch_normalization_3/beta/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/dense/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_1/kernel/mCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_1/bias/mCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ё~
VARIABLE_VALUE(Adam/convolutional_block/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUE&Adam/convolutional_block/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Љі
VARIABLE_VALUE4Adam/convolutional_block/batch_normalization/gamma/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
љЅ
VARIABLE_VALUE3Adam/convolutional_block/batch_normalization/beta/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ѕѓ
VARIABLE_VALUE,Adam/convolutional_block_1/conv2d_1/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Єђ
VARIABLE_VALUE*Adam/convolutional_block_1/conv2d_1/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ћј
VARIABLE_VALUE8Adam/convolutional_block_1/batch_normalization_1/gamma/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ћЇ
VARIABLE_VALUE7Adam/convolutional_block_1/batch_normalization_1/beta/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
іЃ
VARIABLE_VALUE,Adam/convolutional_block_2/conv2d_2/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ѕЂ
VARIABLE_VALUE*Adam/convolutional_block_2/conv2d_2/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ќЈ
VARIABLE_VALUE8Adam/convolutional_block_2/batch_normalization_2/gamma/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ћј
VARIABLE_VALUE7Adam/convolutional_block_2/batch_normalization_2/beta/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
іЃ
VARIABLE_VALUE,Adam/convolutional_block_3/conv2d_3/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ѕЂ
VARIABLE_VALUE*Adam/convolutional_block_3/conv2d_3/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ќЈ
VARIABLE_VALUE8Adam/convolutional_block_3/batch_normalization_3/gamma/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ћј
VARIABLE_VALUE7Adam/convolutional_block_3/batch_normalization_3/beta/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/dense/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/dense_1/kernel/vCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/dense_1/bias/vCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
 &
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename5convolutional_block/conv2d/kernel/Read/ReadVariableOp3convolutional_block/conv2d/bias/Read/ReadVariableOpAconvolutional_block/batch_normalization/gamma/Read/ReadVariableOp@convolutional_block/batch_normalization/beta/Read/ReadVariableOpGconvolutional_block/batch_normalization/moving_mean/Read/ReadVariableOpKconvolutional_block/batch_normalization/moving_variance/Read/ReadVariableOp9convolutional_block_1/conv2d_1/kernel/Read/ReadVariableOp7convolutional_block_1/conv2d_1/bias/Read/ReadVariableOpEconvolutional_block_1/batch_normalization_1/gamma/Read/ReadVariableOpDconvolutional_block_1/batch_normalization_1/beta/Read/ReadVariableOpKconvolutional_block_1/batch_normalization_1/moving_mean/Read/ReadVariableOpOconvolutional_block_1/batch_normalization_1/moving_variance/Read/ReadVariableOp9convolutional_block_2/conv2d_2/kernel/Read/ReadVariableOp7convolutional_block_2/conv2d_2/bias/Read/ReadVariableOpEconvolutional_block_2/batch_normalization_2/gamma/Read/ReadVariableOpDconvolutional_block_2/batch_normalization_2/beta/Read/ReadVariableOpKconvolutional_block_2/batch_normalization_2/moving_mean/Read/ReadVariableOpOconvolutional_block_2/batch_normalization_2/moving_variance/Read/ReadVariableOp9convolutional_block_3/conv2d_3/kernel/Read/ReadVariableOp7convolutional_block_3/conv2d_3/bias/Read/ReadVariableOpEconvolutional_block_3/batch_normalization_3/gamma/Read/ReadVariableOpDconvolutional_block_3/batch_normalization_3/beta/Read/ReadVariableOpKconvolutional_block_3/batch_normalization_3/moving_mean/Read/ReadVariableOpOconvolutional_block_3/batch_normalization_3/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp<Adam/convolutional_block/conv2d/kernel/m/Read/ReadVariableOp:Adam/convolutional_block/conv2d/bias/m/Read/ReadVariableOpHAdam/convolutional_block/batch_normalization/gamma/m/Read/ReadVariableOpGAdam/convolutional_block/batch_normalization/beta/m/Read/ReadVariableOp@Adam/convolutional_block_1/conv2d_1/kernel/m/Read/ReadVariableOp>Adam/convolutional_block_1/conv2d_1/bias/m/Read/ReadVariableOpLAdam/convolutional_block_1/batch_normalization_1/gamma/m/Read/ReadVariableOpKAdam/convolutional_block_1/batch_normalization_1/beta/m/Read/ReadVariableOp@Adam/convolutional_block_2/conv2d_2/kernel/m/Read/ReadVariableOp>Adam/convolutional_block_2/conv2d_2/bias/m/Read/ReadVariableOpLAdam/convolutional_block_2/batch_normalization_2/gamma/m/Read/ReadVariableOpKAdam/convolutional_block_2/batch_normalization_2/beta/m/Read/ReadVariableOp@Adam/convolutional_block_3/conv2d_3/kernel/m/Read/ReadVariableOp>Adam/convolutional_block_3/conv2d_3/bias/m/Read/ReadVariableOpLAdam/convolutional_block_3/batch_normalization_3/gamma/m/Read/ReadVariableOpKAdam/convolutional_block_3/batch_normalization_3/beta/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp<Adam/convolutional_block/conv2d/kernel/v/Read/ReadVariableOp:Adam/convolutional_block/conv2d/bias/v/Read/ReadVariableOpHAdam/convolutional_block/batch_normalization/gamma/v/Read/ReadVariableOpGAdam/convolutional_block/batch_normalization/beta/v/Read/ReadVariableOp@Adam/convolutional_block_1/conv2d_1/kernel/v/Read/ReadVariableOp>Adam/convolutional_block_1/conv2d_1/bias/v/Read/ReadVariableOpLAdam/convolutional_block_1/batch_normalization_1/gamma/v/Read/ReadVariableOpKAdam/convolutional_block_1/batch_normalization_1/beta/v/Read/ReadVariableOp@Adam/convolutional_block_2/conv2d_2/kernel/v/Read/ReadVariableOp>Adam/convolutional_block_2/conv2d_2/bias/v/Read/ReadVariableOpLAdam/convolutional_block_2/batch_normalization_2/gamma/v/Read/ReadVariableOpKAdam/convolutional_block_2/batch_normalization_2/beta/v/Read/ReadVariableOp@Adam/convolutional_block_3/conv2d_3/kernel/v/Read/ReadVariableOp>Adam/convolutional_block_3/conv2d_3/bias/v/Read/ReadVariableOpLAdam/convolutional_block_3/batch_normalization_3/gamma/v/Read/ReadVariableOpKAdam/convolutional_block_3/batch_normalization_3/beta/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOpConst*Z
TinS
Q2O	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *'
f"R 
__inference__traced_save_68533
Ш
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!convolutional_block/conv2d/kernelconvolutional_block/conv2d/bias-convolutional_block/batch_normalization/gamma,convolutional_block/batch_normalization/beta3convolutional_block/batch_normalization/moving_mean7convolutional_block/batch_normalization/moving_variance%convolutional_block_1/conv2d_1/kernel#convolutional_block_1/conv2d_1/bias1convolutional_block_1/batch_normalization_1/gamma0convolutional_block_1/batch_normalization_1/beta7convolutional_block_1/batch_normalization_1/moving_mean;convolutional_block_1/batch_normalization_1/moving_variance%convolutional_block_2/conv2d_2/kernel#convolutional_block_2/conv2d_2/bias1convolutional_block_2/batch_normalization_2/gamma0convolutional_block_2/batch_normalization_2/beta7convolutional_block_2/batch_normalization_2/moving_mean;convolutional_block_2/batch_normalization_2/moving_variance%convolutional_block_3/conv2d_3/kernel#convolutional_block_3/conv2d_3/bias1convolutional_block_3/batch_normalization_3/gamma0convolutional_block_3/batch_normalization_3/beta7convolutional_block_3/batch_normalization_3/moving_mean;convolutional_block_3/batch_normalization_3/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcount(Adam/convolutional_block/conv2d/kernel/m&Adam/convolutional_block/conv2d/bias/m4Adam/convolutional_block/batch_normalization/gamma/m3Adam/convolutional_block/batch_normalization/beta/m,Adam/convolutional_block_1/conv2d_1/kernel/m*Adam/convolutional_block_1/conv2d_1/bias/m8Adam/convolutional_block_1/batch_normalization_1/gamma/m7Adam/convolutional_block_1/batch_normalization_1/beta/m,Adam/convolutional_block_2/conv2d_2/kernel/m*Adam/convolutional_block_2/conv2d_2/bias/m8Adam/convolutional_block_2/batch_normalization_2/gamma/m7Adam/convolutional_block_2/batch_normalization_2/beta/m,Adam/convolutional_block_3/conv2d_3/kernel/m*Adam/convolutional_block_3/conv2d_3/bias/m8Adam/convolutional_block_3/batch_normalization_3/gamma/m7Adam/convolutional_block_3/batch_normalization_3/beta/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/m(Adam/convolutional_block/conv2d/kernel/v&Adam/convolutional_block/conv2d/bias/v4Adam/convolutional_block/batch_normalization/gamma/v3Adam/convolutional_block/batch_normalization/beta/v,Adam/convolutional_block_1/conv2d_1/kernel/v*Adam/convolutional_block_1/conv2d_1/bias/v8Adam/convolutional_block_1/batch_normalization_1/gamma/v7Adam/convolutional_block_1/batch_normalization_1/beta/v,Adam/convolutional_block_2/conv2d_2/kernel/v*Adam/convolutional_block_2/conv2d_2/bias/v8Adam/convolutional_block_2/batch_normalization_2/gamma/v7Adam/convolutional_block_2/batch_normalization_2/beta/v,Adam/convolutional_block_3/conv2d_3/kernel/v*Adam/convolutional_block_3/conv2d_3/bias/v8Adam/convolutional_block_3/batch_normalization_3/gamma/v7Adam/convolutional_block_3/batch_normalization_3/beta/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/v*Y
TinR
P2N*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__traced_restore_68774Ђ╔
╔└
д;
!__inference__traced_restore_68774
file_prefixL
2assignvariableop_convolutional_block_conv2d_kernel: @
2assignvariableop_1_convolutional_block_conv2d_bias: N
@assignvariableop_2_convolutional_block_batch_normalization_gamma: M
?assignvariableop_3_convolutional_block_batch_normalization_beta: T
Fassignvariableop_4_convolutional_block_batch_normalization_moving_mean: X
Jassignvariableop_5_convolutional_block_batch_normalization_moving_variance: R
8assignvariableop_6_convolutional_block_1_conv2d_1_kernel:  D
6assignvariableop_7_convolutional_block_1_conv2d_1_bias: R
Dassignvariableop_8_convolutional_block_1_batch_normalization_1_gamma: Q
Cassignvariableop_9_convolutional_block_1_batch_normalization_1_beta: Y
Kassignvariableop_10_convolutional_block_1_batch_normalization_1_moving_mean: ]
Oassignvariableop_11_convolutional_block_1_batch_normalization_1_moving_variance: S
9assignvariableop_12_convolutional_block_2_conv2d_2_kernel: @E
7assignvariableop_13_convolutional_block_2_conv2d_2_bias:@S
Eassignvariableop_14_convolutional_block_2_batch_normalization_2_gamma:@R
Dassignvariableop_15_convolutional_block_2_batch_normalization_2_beta:@Y
Kassignvariableop_16_convolutional_block_2_batch_normalization_2_moving_mean:@]
Oassignvariableop_17_convolutional_block_2_batch_normalization_2_moving_variance:@S
9assignvariableop_18_convolutional_block_3_conv2d_3_kernel:@@E
7assignvariableop_19_convolutional_block_3_conv2d_3_bias:@S
Eassignvariableop_20_convolutional_block_3_batch_normalization_3_gamma:@R
Dassignvariableop_21_convolutional_block_3_batch_normalization_3_beta:@Y
Kassignvariableop_22_convolutional_block_3_batch_normalization_3_moving_mean:@]
Oassignvariableop_23_convolutional_block_3_batch_normalization_3_moving_variance:@3
 assignvariableop_24_dense_kernel:	@ђ-
assignvariableop_25_dense_bias:	ђ5
"assignvariableop_26_dense_1_kernel:	ђ.
 assignvariableop_27_dense_1_bias:'
assignvariableop_28_adam_iter:	 )
assignvariableop_29_adam_beta_1: )
assignvariableop_30_adam_beta_2: (
assignvariableop_31_adam_decay: 0
&assignvariableop_32_adam_learning_rate: %
assignvariableop_33_total_1: %
assignvariableop_34_count_1: #
assignvariableop_35_total: #
assignvariableop_36_count: V
<assignvariableop_37_adam_convolutional_block_conv2d_kernel_m: H
:assignvariableop_38_adam_convolutional_block_conv2d_bias_m: V
Hassignvariableop_39_adam_convolutional_block_batch_normalization_gamma_m: U
Gassignvariableop_40_adam_convolutional_block_batch_normalization_beta_m: Z
@assignvariableop_41_adam_convolutional_block_1_conv2d_1_kernel_m:  L
>assignvariableop_42_adam_convolutional_block_1_conv2d_1_bias_m: Z
Lassignvariableop_43_adam_convolutional_block_1_batch_normalization_1_gamma_m: Y
Kassignvariableop_44_adam_convolutional_block_1_batch_normalization_1_beta_m: Z
@assignvariableop_45_adam_convolutional_block_2_conv2d_2_kernel_m: @L
>assignvariableop_46_adam_convolutional_block_2_conv2d_2_bias_m:@Z
Lassignvariableop_47_adam_convolutional_block_2_batch_normalization_2_gamma_m:@Y
Kassignvariableop_48_adam_convolutional_block_2_batch_normalization_2_beta_m:@Z
@assignvariableop_49_adam_convolutional_block_3_conv2d_3_kernel_m:@@L
>assignvariableop_50_adam_convolutional_block_3_conv2d_3_bias_m:@Z
Lassignvariableop_51_adam_convolutional_block_3_batch_normalization_3_gamma_m:@Y
Kassignvariableop_52_adam_convolutional_block_3_batch_normalization_3_beta_m:@:
'assignvariableop_53_adam_dense_kernel_m:	@ђ4
%assignvariableop_54_adam_dense_bias_m:	ђ<
)assignvariableop_55_adam_dense_1_kernel_m:	ђ5
'assignvariableop_56_adam_dense_1_bias_m:V
<assignvariableop_57_adam_convolutional_block_conv2d_kernel_v: H
:assignvariableop_58_adam_convolutional_block_conv2d_bias_v: V
Hassignvariableop_59_adam_convolutional_block_batch_normalization_gamma_v: U
Gassignvariableop_60_adam_convolutional_block_batch_normalization_beta_v: Z
@assignvariableop_61_adam_convolutional_block_1_conv2d_1_kernel_v:  L
>assignvariableop_62_adam_convolutional_block_1_conv2d_1_bias_v: Z
Lassignvariableop_63_adam_convolutional_block_1_batch_normalization_1_gamma_v: Y
Kassignvariableop_64_adam_convolutional_block_1_batch_normalization_1_beta_v: Z
@assignvariableop_65_adam_convolutional_block_2_conv2d_2_kernel_v: @L
>assignvariableop_66_adam_convolutional_block_2_conv2d_2_bias_v:@Z
Lassignvariableop_67_adam_convolutional_block_2_batch_normalization_2_gamma_v:@Y
Kassignvariableop_68_adam_convolutional_block_2_batch_normalization_2_beta_v:@Z
@assignvariableop_69_adam_convolutional_block_3_conv2d_3_kernel_v:@@L
>assignvariableop_70_adam_convolutional_block_3_conv2d_3_bias_v:@Z
Lassignvariableop_71_adam_convolutional_block_3_batch_normalization_3_gamma_v:@Y
Kassignvariableop_72_adam_convolutional_block_3_batch_normalization_3_beta_v:@:
'assignvariableop_73_adam_dense_kernel_v:	@ђ4
%assignvariableop_74_adam_dense_bias_v:	ђ<
)assignvariableop_75_adam_dense_1_kernel_v:	ђ5
'assignvariableop_76_adam_dense_1_bias_v:
identity_78ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_39бAssignVariableOp_4бAssignVariableOp_40бAssignVariableOp_41бAssignVariableOp_42бAssignVariableOp_43бAssignVariableOp_44бAssignVariableOp_45бAssignVariableOp_46бAssignVariableOp_47бAssignVariableOp_48бAssignVariableOp_49бAssignVariableOp_5бAssignVariableOp_50бAssignVariableOp_51бAssignVariableOp_52бAssignVariableOp_53бAssignVariableOp_54бAssignVariableOp_55бAssignVariableOp_56бAssignVariableOp_57бAssignVariableOp_58бAssignVariableOp_59бAssignVariableOp_6бAssignVariableOp_60бAssignVariableOp_61бAssignVariableOp_62бAssignVariableOp_63бAssignVariableOp_64бAssignVariableOp_65бAssignVariableOp_66бAssignVariableOp_67бAssignVariableOp_68бAssignVariableOp_69бAssignVariableOp_7бAssignVariableOp_70бAssignVariableOp_71бAssignVariableOp_72бAssignVariableOp_73бAssignVariableOp_74бAssignVariableOp_75бAssignVariableOp_76бAssignVariableOp_8бAssignVariableOp_9З"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*џ"
valueљ"BЇ"NB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЈ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*▒
valueДBцNB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Д
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╬
_output_shapes╗
И::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*\
dtypesR
P2N	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOpAssignVariableOp2assignvariableop_convolutional_block_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_1AssignVariableOp2assignvariableop_1_convolutional_block_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_2AssignVariableOp@assignvariableop_2_convolutional_block_batch_normalization_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_3AssignVariableOp?assignvariableop_3_convolutional_block_batch_normalization_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:х
AssignVariableOp_4AssignVariableOpFassignvariableop_4_convolutional_block_batch_normalization_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_5AssignVariableOpJassignvariableop_5_convolutional_block_batch_normalization_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_6AssignVariableOp8assignvariableop_6_convolutional_block_1_conv2d_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_7AssignVariableOp6assignvariableop_7_convolutional_block_1_conv2d_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:│
AssignVariableOp_8AssignVariableOpDassignvariableop_8_convolutional_block_1_batch_normalization_1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:▓
AssignVariableOp_9AssignVariableOpCassignvariableop_9_convolutional_block_1_batch_normalization_1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_10AssignVariableOpKassignvariableop_10_convolutional_block_1_batch_normalization_1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_11AssignVariableOpOassignvariableop_11_convolutional_block_1_batch_normalization_1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_12AssignVariableOp9assignvariableop_12_convolutional_block_2_conv2d_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_13AssignVariableOp7assignvariableop_13_convolutional_block_2_conv2d_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_14AssignVariableOpEassignvariableop_14_convolutional_block_2_batch_normalization_2_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:х
AssignVariableOp_15AssignVariableOpDassignvariableop_15_convolutional_block_2_batch_normalization_2_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_16AssignVariableOpKassignvariableop_16_convolutional_block_2_batch_normalization_2_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_17AssignVariableOpOassignvariableop_17_convolutional_block_2_batch_normalization_2_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_18AssignVariableOp9assignvariableop_18_convolutional_block_3_conv2d_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_19AssignVariableOp7assignvariableop_19_convolutional_block_3_conv2d_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_20AssignVariableOpEassignvariableop_20_convolutional_block_3_batch_normalization_3_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:х
AssignVariableOp_21AssignVariableOpDassignvariableop_21_convolutional_block_3_batch_normalization_3_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_22AssignVariableOpKassignvariableop_22_convolutional_block_3_batch_normalization_3_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_23AssignVariableOpOassignvariableop_23_convolutional_block_3_batch_normalization_3_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_24AssignVariableOp assignvariableop_24_dense_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_25AssignVariableOpassignvariableop_25_dense_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Њ
AssignVariableOp_26AssignVariableOp"assignvariableop_26_dense_1_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_27AssignVariableOp assignvariableop_27_dense_1_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0	*
_output_shapes
:ј
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_iterIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_29AssignVariableOpassignvariableop_29_adam_beta_1Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:љ
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_beta_2Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_decayIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_32AssignVariableOp&assignvariableop_32_adam_learning_rateIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_33AssignVariableOpassignvariableop_33_total_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_34AssignVariableOpassignvariableop_34_count_1Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_37AssignVariableOp<assignvariableop_37_adam_convolutional_block_conv2d_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_38AssignVariableOp:assignvariableop_38_adam_convolutional_block_conv2d_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_39AssignVariableOpHassignvariableop_39_adam_convolutional_block_batch_normalization_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_40AssignVariableOpGassignvariableop_40_adam_convolutional_block_batch_normalization_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_41AssignVariableOp@assignvariableop_41_adam_convolutional_block_1_conv2d_1_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_42AssignVariableOp>assignvariableop_42_adam_convolutional_block_1_conv2d_1_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_43AssignVariableOpLassignvariableop_43_adam_convolutional_block_1_batch_normalization_1_gamma_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_44AssignVariableOpKassignvariableop_44_adam_convolutional_block_1_batch_normalization_1_beta_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_45AssignVariableOp@assignvariableop_45_adam_convolutional_block_2_conv2d_2_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_46AssignVariableOp>assignvariableop_46_adam_convolutional_block_2_conv2d_2_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_47AssignVariableOpLassignvariableop_47_adam_convolutional_block_2_batch_normalization_2_gamma_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_48AssignVariableOpKassignvariableop_48_adam_convolutional_block_2_batch_normalization_2_beta_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_49AssignVariableOp@assignvariableop_49_adam_convolutional_block_3_conv2d_3_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_50AssignVariableOp>assignvariableop_50_adam_convolutional_block_3_conv2d_3_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_51AssignVariableOpLassignvariableop_51_adam_convolutional_block_3_batch_normalization_3_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_52AssignVariableOpKassignvariableop_52_adam_convolutional_block_3_batch_normalization_3_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_53AssignVariableOp'assignvariableop_53_adam_dense_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_54AssignVariableOp%assignvariableop_54_adam_dense_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_dense_1_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_dense_1_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:Г
AssignVariableOp_57AssignVariableOp<assignvariableop_57_adam_convolutional_block_conv2d_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:Ф
AssignVariableOp_58AssignVariableOp:assignvariableop_58_adam_convolutional_block_conv2d_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:╣
AssignVariableOp_59AssignVariableOpHassignvariableop_59_adam_convolutional_block_batch_normalization_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_60AssignVariableOpGassignvariableop_60_adam_convolutional_block_batch_normalization_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_61AssignVariableOp@assignvariableop_61_adam_convolutional_block_1_conv2d_1_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_62AssignVariableOp>assignvariableop_62_adam_convolutional_block_1_conv2d_1_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_63AssignVariableOpLassignvariableop_63_adam_convolutional_block_1_batch_normalization_1_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_64AssignVariableOpKassignvariableop_64_adam_convolutional_block_1_batch_normalization_1_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_65AssignVariableOp@assignvariableop_65_adam_convolutional_block_2_conv2d_2_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_66AssignVariableOp>assignvariableop_66_adam_convolutional_block_2_conv2d_2_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_67AssignVariableOpLassignvariableop_67_adam_convolutional_block_2_batch_normalization_2_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_68AssignVariableOpKassignvariableop_68_adam_convolutional_block_2_batch_normalization_2_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_69AssignVariableOp@assignvariableop_69_adam_convolutional_block_3_conv2d_3_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_70AssignVariableOp>assignvariableop_70_adam_convolutional_block_3_conv2d_3_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_71AssignVariableOpLassignvariableop_71_adam_convolutional_block_3_batch_normalization_3_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_72AssignVariableOpKassignvariableop_72_adam_convolutional_block_3_batch_normalization_3_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_73AssignVariableOp'assignvariableop_73_adam_dense_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:ќ
AssignVariableOp_74AssignVariableOp%assignvariableop_74_adam_dense_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_75AssignVariableOp)assignvariableop_75_adam_dense_1_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_76AssignVariableOp'assignvariableop_76_adam_dense_1_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ь
Identity_77Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_78IdentityIdentity_77:output:0^NoOp_1*
T0*
_output_shapes
: ┌
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_78Identity_78:output:0*▒
_input_shapesЪ
ю: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╦
Џ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68170

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ђ
Ѕ
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_67822

inputsA
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@;
-batch_normalization_3_readvariableop_resource:@=
/batch_normalization_3_readvariableop_1_resource:@L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1бconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOpј
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0г
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
ё
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ў
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ф
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
ј
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0њ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Й
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_3/MaxPool:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( r
ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @╦
NoOpNoOp6^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @: : : : : : 2n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
њ
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_68126

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
»

Щ
A__inference_conv2d_layer_call_and_return_conditional_losses_64907

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ю
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ■■ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
ё

a
B__inference_dropout_layer_call_and_return_conditional_losses_66121

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ў
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>Д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
я	
Ъ
3__inference_convolutional_block_layer_call_fn_64940
convolutional_block_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identityѕбStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallconvolutional_block_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_convolutional_block_layer_call_and_return_conditional_losses_64925w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
1
_output_shapes
:         ђђ
3
_user_specified_nameconvolutional_block_input
ј	
╬
3__inference_batch_normalization_layer_call_fn_67957

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_64848Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
┬
Х
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_65392

inputs(
conv2d_2_65375: @
conv2d_2_65377:@)
batch_normalization_2_65381:@)
batch_normalization_2_65383:@)
batch_normalization_2_65385:@)
batch_normalization_2_65387:@
identityѕб-batch_normalization_2/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallЭ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_65375conv2d_2_65377*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_65327з
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_65243і
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_65381batch_normalization_2_65383batch_normalization_2_65385batch_normalization_2_65387*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_65299~
ReluRelu6batch_normalization_2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @Ў
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         >> : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:         >> 
 
_user_specified_nameinputs
ї%
╬

E__inference_sequential_layer_call_and_return_conditional_losses_65802

inputs3
convolutional_block_65749: '
convolutional_block_65751: '
convolutional_block_65753: '
convolutional_block_65755: '
convolutional_block_65757: '
convolutional_block_65759: 5
convolutional_block_1_65762:  )
convolutional_block_1_65764: )
convolutional_block_1_65766: )
convolutional_block_1_65768: )
convolutional_block_1_65770: )
convolutional_block_1_65772: 5
convolutional_block_2_65775: @)
convolutional_block_2_65777:@)
convolutional_block_2_65779:@)
convolutional_block_2_65781:@)
convolutional_block_2_65783:@)
convolutional_block_2_65785:@5
convolutional_block_3_65788:@@)
convolutional_block_3_65790:@)
convolutional_block_3_65792:@)
convolutional_block_3_65794:@)
convolutional_block_3_65796:@)
convolutional_block_3_65798:@
identityѕб+convolutional_block/StatefulPartitionedCallб-convolutional_block_1/StatefulPartitionedCallб-convolutional_block_2/StatefulPartitionedCallб-convolutional_block_3/StatefulPartitionedCallќ
+convolutional_block/StatefulPartitionedCallStatefulPartitionedCallinputsconvolutional_block_65749convolutional_block_65751convolutional_block_65753convolutional_block_65755convolutional_block_65757convolutional_block_65759*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_convolutional_block_layer_call_and_return_conditional_losses_64989н
-convolutional_block_1/StatefulPartitionedCallStatefulPartitionedCall4convolutional_block/StatefulPartitionedCall:output:0convolutional_block_1_65762convolutional_block_1_65764convolutional_block_1_65766convolutional_block_1_65768convolutional_block_1_65770convolutional_block_1_65772*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >> *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_65219о
-convolutional_block_2/StatefulPartitionedCallStatefulPartitionedCall6convolutional_block_1/StatefulPartitionedCall:output:0convolutional_block_2_65775convolutional_block_2_65777convolutional_block_2_65779convolutional_block_2_65781convolutional_block_2_65783convolutional_block_2_65785*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_65392о
-convolutional_block_3/StatefulPartitionedCallStatefulPartitionedCall6convolutional_block_2/StatefulPartitionedCall:output:0convolutional_block_3_65788convolutional_block_3_65790convolutional_block_3_65792convolutional_block_3_65794convolutional_block_3_65796convolutional_block_3_65798*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_65565Ї
IdentityIdentity6convolutional_block_3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @ё
NoOpNoOp,^convolutional_block/StatefulPartitionedCall.^convolutional_block_1/StatefulPartitionedCall.^convolutional_block_2/StatefulPartitionedCall.^convolutional_block_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+convolutional_block/StatefulPartitionedCall+convolutional_block/StatefulPartitionedCall2^
-convolutional_block_1/StatefulPartitionedCall-convolutional_block_1/StatefulPartitionedCall2^
-convolutional_block_2/StatefulPartitionedCall-convolutional_block_2/StatefulPartitionedCall2^
-convolutional_block_3/StatefulPartitionedCall-convolutional_block_3/StatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
№
Џ
&__inference_conv2d_layer_call_fn_67924

inputs!
unknown: 
	unknown_0: 
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ■■ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_64907y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:         ■■ `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
њ
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_68217

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ъ

з
@__inference_dense_layer_call_and_return_conditional_losses_67868

inputs1
matmul_readvariableop_resource:	@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
▒
О
,__inference_sequential_1_layer_call_fn_66091
dense_input
unknown:	@ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:
identityѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_66080o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:         @
%
_user_specified_namedense_input
─
Ћ
'__inference_dense_1_layer_call_fn_67904

inputs
unknown:	ђ
	unknown_0:
identityѕбStatefulPartitionedCall┌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_66073o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
┼
и
N__inference_convolutional_block_layer_call_and_return_conditional_losses_65061
convolutional_block_input&
conv2d_65044: 
conv2d_65046: '
batch_normalization_65050: '
batch_normalization_65052: '
batch_normalization_65054: '
batch_normalization_65056: 
identityѕб+batch_normalization/StatefulPartitionedCallбconv2d/StatefulPartitionedCallЁ
conv2d/StatefulPartitionedCallStatefulPartitionedCallconvolutional_block_inputconv2d_65044conv2d_65046*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ■■ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_64907ь
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_64823Ч
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_65050batch_normalization_65052batch_normalization_65054batch_normalization_65056*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_64879|
ReluRelu4batch_normalization/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          Ћ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:l h
1
_output_shapes
:         ђђ
3
_user_specified_nameconvolutional_block_input
ИЭ
▄)
N__inference_convolutional_model_layer_call_and_return_conditional_losses_67132

inputs^
Dsequential_convolutional_block_conv2d_conv2d_readvariableop_resource: S
Esequential_convolutional_block_conv2d_biasadd_readvariableop_resource: X
Jsequential_convolutional_block_batch_normalization_readvariableop_resource: Z
Lsequential_convolutional_block_batch_normalization_readvariableop_1_resource: i
[sequential_convolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_resource: k
]sequential_convolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: b
Hsequential_convolutional_block_1_conv2d_1_conv2d_readvariableop_resource:  W
Isequential_convolutional_block_1_conv2d_1_biasadd_readvariableop_resource: \
Nsequential_convolutional_block_1_batch_normalization_1_readvariableop_resource: ^
Psequential_convolutional_block_1_batch_normalization_1_readvariableop_1_resource: m
_sequential_convolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource: o
asequential_convolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: b
Hsequential_convolutional_block_2_conv2d_2_conv2d_readvariableop_resource: @W
Isequential_convolutional_block_2_conv2d_2_biasadd_readvariableop_resource:@\
Nsequential_convolutional_block_2_batch_normalization_2_readvariableop_resource:@^
Psequential_convolutional_block_2_batch_normalization_2_readvariableop_1_resource:@m
_sequential_convolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@o
asequential_convolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@b
Hsequential_convolutional_block_3_conv2d_3_conv2d_readvariableop_resource:@@W
Isequential_convolutional_block_3_conv2d_3_biasadd_readvariableop_resource:@\
Nsequential_convolutional_block_3_batch_normalization_3_readvariableop_resource:@^
Psequential_convolutional_block_3_batch_normalization_3_readvariableop_1_resource:@m
_sequential_convolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@o
asequential_convolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@D
1sequential_1_dense_matmul_readvariableop_resource:	@ђA
2sequential_1_dense_biasadd_readvariableop_resource:	ђF
3sequential_1_dense_1_matmul_readvariableop_resource:	ђB
4sequential_1_dense_1_biasadd_readvariableop_resource:
identityѕбAsequential/convolutional_block/batch_normalization/AssignNewValueбCsequential/convolutional_block/batch_normalization/AssignNewValue_1бRsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpбTsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1бAsequential/convolutional_block/batch_normalization/ReadVariableOpбCsequential/convolutional_block/batch_normalization/ReadVariableOp_1б<sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOpб;sequential/convolutional_block/conv2d/Conv2D/ReadVariableOpбEsequential/convolutional_block_1/batch_normalization_1/AssignNewValueбGsequential/convolutional_block_1/batch_normalization_1/AssignNewValue_1бVsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpбXsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1бEsequential/convolutional_block_1/batch_normalization_1/ReadVariableOpбGsequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_1б@sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOpб?sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOpбEsequential/convolutional_block_2/batch_normalization_2/AssignNewValueбGsequential/convolutional_block_2/batch_normalization_2/AssignNewValue_1бVsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpбXsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1бEsequential/convolutional_block_2/batch_normalization_2/ReadVariableOpбGsequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_1б@sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOpб?sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOpбEsequential/convolutional_block_3/batch_normalization_3/AssignNewValueбGsequential/convolutional_block_3/batch_normalization_3/AssignNewValue_1бVsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpбXsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1бEsequential/convolutional_block_3/batch_normalization_3/ReadVariableOpбGsequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_1б@sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOpб?sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOpб)sequential_1/dense/BiasAdd/ReadVariableOpб(sequential_1/dense/MatMul/ReadVariableOpб+sequential_1/dense_1/BiasAdd/ReadVariableOpб*sequential_1/dense_1/MatMul/ReadVariableOp╚
;sequential/convolutional_block/conv2d/Conv2D/ReadVariableOpReadVariableOpDsequential_convolutional_block_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0У
,sequential/convolutional_block/conv2d/Conv2DConv2DinputsCsequential/convolutional_block/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ *
paddingVALID*
strides
Й
<sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOpReadVariableOpEsequential_convolutional_block_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ы
-sequential/convolutional_block/conv2d/BiasAddBiasAdd5sequential/convolutional_block/conv2d/Conv2D:output:0Dsequential/convolutional_block/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ С
4sequential/convolutional_block/max_pooling2d/MaxPoolMaxPool6sequential/convolutional_block/conv2d/BiasAdd:output:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
╚
Asequential/convolutional_block/batch_normalization/ReadVariableOpReadVariableOpJsequential_convolutional_block_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0╠
Csequential/convolutional_block/batch_normalization/ReadVariableOp_1ReadVariableOpLsequential_convolutional_block_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0Ж
Rsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp[sequential_convolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ь
Tsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]sequential_convolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Щ
Csequential/convolutional_block/batch_normalization/FusedBatchNormV3FusedBatchNormV3=sequential/convolutional_block/max_pooling2d/MaxPool:output:0Isequential/convolutional_block/batch_normalization/ReadVariableOp:value:0Ksequential/convolutional_block/batch_normalization/ReadVariableOp_1:value:0Zsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0\sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<њ
Asequential/convolutional_block/batch_normalization/AssignNewValueAssignVariableOp[sequential_convolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_resourcePsequential/convolutional_block/batch_normalization/FusedBatchNormV3:batch_mean:0S^sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ю
Csequential/convolutional_block/batch_normalization/AssignNewValue_1AssignVariableOp]sequential_convolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceTsequential/convolutional_block/batch_normalization/FusedBatchNormV3:batch_variance:0U^sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(«
#sequential/convolutional_block/ReluReluGsequential/convolutional_block/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          л
?sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpHsequential_convolutional_block_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Ў
0sequential/convolutional_block_1/conv2d_1/Conv2DConv2D1sequential/convolutional_block/Relu:activations:0Gsequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} *
paddingVALID*
strides
к
@sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpIsequential_convolutional_block_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ч
1sequential/convolutional_block_1/conv2d_1/BiasAddBiasAdd9sequential/convolutional_block_1/conv2d_1/Conv2D:output:0Hsequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} В
8sequential/convolutional_block_1/max_pooling2d_1/MaxPoolMaxPool:sequential/convolutional_block_1/conv2d_1/BiasAdd:output:0*/
_output_shapes
:         >> *
ksize
*
paddingVALID*
strides
л
Esequential/convolutional_block_1/batch_normalization_1/ReadVariableOpReadVariableOpNsequential_convolutional_block_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype0н
Gsequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_1ReadVariableOpPsequential_convolutional_block_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype0Ы
Vsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp_sequential_convolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ш
Xsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpasequential_convolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0њ
Gsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3Asequential/convolutional_block_1/max_pooling2d_1/MaxPool:output:0Msequential/convolutional_block_1/batch_normalization_1/ReadVariableOp:value:0Osequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_1:value:0^sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0`sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         >> : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
Esequential/convolutional_block_1/batch_normalization_1/AssignNewValueAssignVariableOp_sequential_convolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceTsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3:batch_mean:0W^sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
Gsequential/convolutional_block_1/batch_normalization_1/AssignNewValue_1AssignVariableOpasequential_convolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceXsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3:batch_variance:0Y^sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(┤
%sequential/convolutional_block_1/ReluReluKsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         >> л
?sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOpHsequential_convolutional_block_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Џ
0sequential/convolutional_block_2/conv2d_2/Conv2DConv2D3sequential/convolutional_block_1/Relu:activations:0Gsequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@*
paddingVALID*
strides
к
@sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpIsequential_convolutional_block_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ч
1sequential/convolutional_block_2/conv2d_2/BiasAddBiasAdd9sequential/convolutional_block_2/conv2d_2/Conv2D:output:0Hsequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@В
8sequential/convolutional_block_2/max_pooling2d_2/MaxPoolMaxPool:sequential/convolutional_block_2/conv2d_2/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
л
Esequential/convolutional_block_2/batch_normalization_2/ReadVariableOpReadVariableOpNsequential_convolutional_block_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0н
Gsequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_1ReadVariableOpPsequential_convolutional_block_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ы
Vsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp_sequential_convolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
Xsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpasequential_convolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0њ
Gsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3Asequential/convolutional_block_2/max_pooling2d_2/MaxPool:output:0Msequential/convolutional_block_2/batch_normalization_2/ReadVariableOp:value:0Osequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_1:value:0^sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0`sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
Esequential/convolutional_block_2/batch_normalization_2/AssignNewValueAssignVariableOp_sequential_convolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceTsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3:batch_mean:0W^sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
Gsequential/convolutional_block_2/batch_normalization_2/AssignNewValue_1AssignVariableOpasequential_convolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceXsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3:batch_variance:0Y^sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(┤
%sequential/convolutional_block_2/ReluReluKsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @л
?sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpHsequential_convolutional_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Џ
0sequential/convolutional_block_3/conv2d_3/Conv2DConv2D3sequential/convolutional_block_2/Relu:activations:0Gsequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
к
@sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpIsequential_convolutional_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ч
1sequential/convolutional_block_3/conv2d_3/BiasAddBiasAdd9sequential/convolutional_block_3/conv2d_3/Conv2D:output:0Hsequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @В
8sequential/convolutional_block_3/max_pooling2d_3/MaxPoolMaxPool:sequential/convolutional_block_3/conv2d_3/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
л
Esequential/convolutional_block_3/batch_normalization_3/ReadVariableOpReadVariableOpNsequential_convolutional_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0н
Gsequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpPsequential_convolutional_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ы
Vsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp_sequential_convolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
Xsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpasequential_convolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0њ
Gsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3Asequential/convolutional_block_3/max_pooling2d_3/MaxPool:output:0Msequential/convolutional_block_3/batch_normalization_3/ReadVariableOp:value:0Osequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_1:value:0^sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0`sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<б
Esequential/convolutional_block_3/batch_normalization_3/AssignNewValueAssignVariableOp_sequential_convolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceTsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3:batch_mean:0W^sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(г
Gsequential/convolutional_block_3/batch_normalization_3/AssignNewValue_1AssignVariableOpasequential_convolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceXsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3:batch_variance:0Y^sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(┤
%sequential/convolutional_block_3/ReluReluKsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @ђ
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      к
global_average_pooling2d/MeanMean3sequential/convolutional_block_3/Relu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @Џ
(sequential_1/dense/MatMul/ReadVariableOpReadVariableOp1sequential_1_dense_matmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0░
sequential_1/dense/MatMulMatMul&global_average_pooling2d/Mean:output:00sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЎ
)sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0░
sequential_1/dense/BiasAddBiasAdd#sequential_1/dense/MatMul:product:01sequential_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђw
sequential_1/dense/ReluRelu#sequential_1/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђg
"sequential_1/dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?«
 sequential_1/dropout/dropout/MulMul%sequential_1/dense/Relu:activations:0+sequential_1/dropout/dropout/Const:output:0*
T0*(
_output_shapes
:         ђw
"sequential_1/dropout/dropout/ShapeShape%sequential_1/dense/Relu:activations:0*
T0*
_output_shapes
:├
9sequential_1/dropout/dropout/random_uniform/RandomUniformRandomUniform+sequential_1/dropout/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*

seedp
+sequential_1/dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>Т
)sequential_1/dropout/dropout/GreaterEqualGreaterEqualBsequential_1/dropout/dropout/random_uniform/RandomUniform:output:04sequential_1/dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђџ
!sequential_1/dropout/dropout/CastCast-sequential_1/dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђЕ
"sequential_1/dropout/dropout/Mul_1Mul$sequential_1/dropout/dropout/Mul:z:0%sequential_1/dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђЪ
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0│
sequential_1/dense_1/MatMulMatMul&sequential_1/dropout/dropout/Mul_1:z:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ю
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0х
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ђ
sequential_1/dense_1/SoftmaxSoftmax%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         u
IdentityIdentity&sequential_1/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ╠
NoOpNoOpB^sequential/convolutional_block/batch_normalization/AssignNewValueD^sequential/convolutional_block/batch_normalization/AssignNewValue_1S^sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpU^sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1B^sequential/convolutional_block/batch_normalization/ReadVariableOpD^sequential/convolutional_block/batch_normalization/ReadVariableOp_1=^sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOp<^sequential/convolutional_block/conv2d/Conv2D/ReadVariableOpF^sequential/convolutional_block_1/batch_normalization_1/AssignNewValueH^sequential/convolutional_block_1/batch_normalization_1/AssignNewValue_1W^sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpY^sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1F^sequential/convolutional_block_1/batch_normalization_1/ReadVariableOpH^sequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_1A^sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp@^sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOpF^sequential/convolutional_block_2/batch_normalization_2/AssignNewValueH^sequential/convolutional_block_2/batch_normalization_2/AssignNewValue_1W^sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpY^sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1F^sequential/convolutional_block_2/batch_normalization_2/ReadVariableOpH^sequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_1A^sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp@^sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOpF^sequential/convolutional_block_3/batch_normalization_3/AssignNewValueH^sequential/convolutional_block_3/batch_normalization_3/AssignNewValue_1W^sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpY^sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1F^sequential/convolutional_block_3/batch_normalization_3/ReadVariableOpH^sequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_1A^sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp@^sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp*^sequential_1/dense/BiasAdd/ReadVariableOp)^sequential_1/dense/MatMul/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2є
Asequential/convolutional_block/batch_normalization/AssignNewValueAsequential/convolutional_block/batch_normalization/AssignNewValue2і
Csequential/convolutional_block/batch_normalization/AssignNewValue_1Csequential/convolutional_block/batch_normalization/AssignNewValue_12е
Rsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpRsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp2г
Tsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Tsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_12є
Asequential/convolutional_block/batch_normalization/ReadVariableOpAsequential/convolutional_block/batch_normalization/ReadVariableOp2і
Csequential/convolutional_block/batch_normalization/ReadVariableOp_1Csequential/convolutional_block/batch_normalization/ReadVariableOp_12|
<sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOp<sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOp2z
;sequential/convolutional_block/conv2d/Conv2D/ReadVariableOp;sequential/convolutional_block/conv2d/Conv2D/ReadVariableOp2ј
Esequential/convolutional_block_1/batch_normalization_1/AssignNewValueEsequential/convolutional_block_1/batch_normalization_1/AssignNewValue2њ
Gsequential/convolutional_block_1/batch_normalization_1/AssignNewValue_1Gsequential/convolutional_block_1/batch_normalization_1/AssignNewValue_12░
Vsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpVsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2┤
Xsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Xsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12ј
Esequential/convolutional_block_1/batch_normalization_1/ReadVariableOpEsequential/convolutional_block_1/batch_normalization_1/ReadVariableOp2њ
Gsequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_1Gsequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_12ё
@sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp@sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp2ѓ
?sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOp?sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOp2ј
Esequential/convolutional_block_2/batch_normalization_2/AssignNewValueEsequential/convolutional_block_2/batch_normalization_2/AssignNewValue2њ
Gsequential/convolutional_block_2/batch_normalization_2/AssignNewValue_1Gsequential/convolutional_block_2/batch_normalization_2/AssignNewValue_12░
Vsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpVsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2┤
Xsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Xsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12ј
Esequential/convolutional_block_2/batch_normalization_2/ReadVariableOpEsequential/convolutional_block_2/batch_normalization_2/ReadVariableOp2њ
Gsequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_1Gsequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_12ё
@sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp@sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp2ѓ
?sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOp?sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOp2ј
Esequential/convolutional_block_3/batch_normalization_3/AssignNewValueEsequential/convolutional_block_3/batch_normalization_3/AssignNewValue2њ
Gsequential/convolutional_block_3/batch_normalization_3/AssignNewValue_1Gsequential/convolutional_block_3/batch_normalization_3/AssignNewValue_12░
Vsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpVsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2┤
Xsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Xsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12ј
Esequential/convolutional_block_3/batch_normalization_3/ReadVariableOpEsequential/convolutional_block_3/batch_normalization_3/ReadVariableOp2њ
Gsequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_1Gsequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_12ё
@sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp@sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp2ѓ
?sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp?sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp2V
)sequential_1/dense/BiasAdd/ReadVariableOp)sequential_1/dense/BiasAdd/ReadVariableOp2T
(sequential_1/dense/MatMul/ReadVariableOp(sequential_1/dense/MatMul/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
┤
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_66028

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
в
Ю
(__inference_conv2d_3_layer_call_fn_68197

inputs!
unknown:@@
	unknown_0:@
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_65500w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╣
K
/__inference_max_pooling2d_1_layer_call_fn_68030

inputs
identity█
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_65070Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
б
м
,__inference_sequential_1_layer_call_fn_67459

inputs
unknown:	@ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_66164o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Ё
┐
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68097

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
┴
ћ
%__inference_dense_layer_call_fn_67857

inputs
unknown:	@ђ
	unknown_0:	ђ
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_66049p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
─
Х
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_65172

inputs(
conv2d_1_65155:  
conv2d_1_65157: )
batch_normalization_1_65161: )
batch_normalization_1_65163: )
batch_normalization_1_65165: )
batch_normalization_1_65167: 
identityѕб-batch_normalization_1/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallЭ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_65155conv2d_1_65157*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         }} *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_65154з
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >> * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_65070ї
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_65161batch_normalization_1_65163batch_normalization_1_65165batch_normalization_1_65167*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >> *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_65095~
ReluRelu6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:         >> i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         >> Ў
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':          : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Ц	
ј
5__inference_convolutional_block_1_layer_call_fn_67607

inputs!
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >> *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_65172w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         >> `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':          : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
м
Ѕ
*__inference_sequential_layer_call_fn_65906
convolutional_block_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityѕбStatefulPartitionedCallЏ
StatefulPartitionedCallStatefulPartitionedCallconvolutional_block_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_65802w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
1
_output_shapes
:         ђђ
3
_user_specified_nameconvolutional_block_input
в
Ю
(__inference_conv2d_1_layer_call_fn_68015

inputs!
unknown:  
	unknown_0: 
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         }} *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_65154w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         }} `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
А
Ш
*__inference_sequential_layer_call_fn_67185

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityѕбStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_65640w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Б	
ј
5__inference_convolutional_block_3_layer_call_fn_67796

inputs!
unknown:@@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identityѕбStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_65565w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╦
Џ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_65095

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
а
C
'__inference_dropout_layer_call_fn_67873

inputs
identity▒
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_66060a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
н
е
G__inference_sequential_1_layer_call_and_return_conditional_losses_66203
dense_input
dense_66191:	@ђ
dense_66193:	ђ 
dense_1_66197:	ђ
dense_1_66199:
identityѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallЖ
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_66191dense_66193*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_66049┘
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_66060є
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_66197dense_1_66199*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_66073w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ѕ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:T P
'
_output_shapes
:         @
%
_user_specified_namedense_input
ї
ц
N__inference_convolutional_block_layer_call_and_return_conditional_losses_64989

inputs&
conv2d_64972: 
conv2d_64974: '
batch_normalization_64978: '
batch_normalization_64980: '
batch_normalization_64982: '
batch_normalization_64984: 
identityѕб+batch_normalization/StatefulPartitionedCallбconv2d/StatefulPartitionedCallЫ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_64972conv2d_64974*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ■■ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_64907ь
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_64823Ч
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_64978batch_normalization_64980batch_normalization_64982batch_normalization_64984*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_64879|
ReluRelu4batch_normalization/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          Ћ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
╣
K
/__inference_max_pooling2d_3_layer_call_fn_68212

inputs
identity█
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_65416Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╔
Ў
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67988

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
љ
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_64823

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ђ
Ѕ
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_67736

inputsA
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб5batch_normalization_2/FusedBatchNormV3/ReadVariableOpб7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_2/ReadVariableOpб&batch_normalization_2/ReadVariableOp_1бconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOpј
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0г
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@*
paddingVALID*
strides
ё
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ў
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@ф
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
ј
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0њ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Й
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_2/MaxPool:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( r
ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @╦
NoOpNoOp6^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         >> : : : : : : 2n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         >> 
 
_user_specified_nameinputs
Д

Ч
C__inference_conv2d_2_layer_call_and_return_conditional_losses_65327

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         <<@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         >> : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         >> 
 
_user_specified_nameinputs
б

З
B__inference_dense_1_layer_call_and_return_conditional_losses_66073

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
─
Х
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_65518

inputs(
conv2d_3_65501:@@
conv2d_3_65503:@)
batch_normalization_3_65507:@)
batch_normalization_3_65509:@)
batch_normalization_3_65511:@)
batch_normalization_3_65513:@
identityѕб-batch_normalization_3/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallЭ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_65501conv2d_3_65503*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_65500з
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_65416ї
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_3_65507batch_normalization_3_65509batch_normalization_3_65511batch_normalization_3_65513*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_65441~
ReluRelu6batch_normalization_3/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @Ў
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @: : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
б

З
B__inference_dense_1_layer_call_and_return_conditional_losses_67915

inputs1
matmul_readvariableop_resource:	ђ-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:         `
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Л
з
3__inference_convolutional_model_layer_call_fn_66590
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	@ђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:
identityѕбStatefulPartitionedCall┬
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_convolutional_model_layer_call_and_return_conditional_losses_66470o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1
Ё
┐
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68188

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╦
Џ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_65441

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
в
Ю
(__inference_conv2d_2_layer_call_fn_68106

inputs!
unknown: @
	unknown_0:@
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_65327w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         <<@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         >> : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         >> 
 
_user_specified_nameinputs
Ъ

з
@__inference_dense_layer_call_and_return_conditional_losses_66049

inputs1
matmul_readvariableop_resource:	@ђ.
biasadd_readvariableop_resource:	ђ
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         ђb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         ђw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╦
Џ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_65268

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
љ	
л
5__inference_batch_normalization_3_layer_call_fn_68243

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_65472Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ќ
T
8__inference_global_average_pooling2d_layer_call_fn_67427

inputs
identity╩
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_66028i
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
љ	
л
5__inference_batch_normalization_1_layer_call_fn_68061

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_65126Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ё
┐
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_65299

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Д
┼
G__inference_sequential_1_layer_call_and_return_conditional_losses_67478

inputs7
$dense_matmul_readvariableop_resource:	@ђ4
%dense_biasadd_readvariableop_resource:	ђ9
&dense_1_matmul_readvariableop_resource:	ђ5
'dense_1_biasadd_readvariableop_resource:
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpЂ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђi
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:         ђЁ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0ї
dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ѓ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ј
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ─
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
њ	
л
5__inference_batch_normalization_1_layer_call_fn_68048

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_65095Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
љ
d
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_67944

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
о
Ы
3__inference_convolutional_model_layer_call_fn_66846

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	@ђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:
identityѕбStatefulPartitionedCall╔
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_convolutional_model_layer_call_and_return_conditional_losses_66285o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
▄	
Ъ
3__inference_convolutional_block_layer_call_fn_65021
convolutional_block_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identityѕбStatefulPartitionedCall│
StatefulPartitionedCallStatefulPartitionedCallconvolutional_block_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_convolutional_block_layer_call_and_return_conditional_losses_64989w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
1
_output_shapes
:         ђђ
3
_user_specified_nameconvolutional_block_input
║Ь
Н-
 __inference__wrapped_model_64814
input_1r
Xconvolutional_model_sequential_convolutional_block_conv2d_conv2d_readvariableop_resource: g
Yconvolutional_model_sequential_convolutional_block_conv2d_biasadd_readvariableop_resource: l
^convolutional_model_sequential_convolutional_block_batch_normalization_readvariableop_resource: n
`convolutional_model_sequential_convolutional_block_batch_normalization_readvariableop_1_resource: }
oconvolutional_model_sequential_convolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_resource: 
qconvolutional_model_sequential_convolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: v
\convolutional_model_sequential_convolutional_block_1_conv2d_1_conv2d_readvariableop_resource:  k
]convolutional_model_sequential_convolutional_block_1_conv2d_1_biasadd_readvariableop_resource: p
bconvolutional_model_sequential_convolutional_block_1_batch_normalization_1_readvariableop_resource: r
dconvolutional_model_sequential_convolutional_block_1_batch_normalization_1_readvariableop_1_resource: Ђ
sconvolutional_model_sequential_convolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource: Ѓ
uconvolutional_model_sequential_convolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: v
\convolutional_model_sequential_convolutional_block_2_conv2d_2_conv2d_readvariableop_resource: @k
]convolutional_model_sequential_convolutional_block_2_conv2d_2_biasadd_readvariableop_resource:@p
bconvolutional_model_sequential_convolutional_block_2_batch_normalization_2_readvariableop_resource:@r
dconvolutional_model_sequential_convolutional_block_2_batch_normalization_2_readvariableop_1_resource:@Ђ
sconvolutional_model_sequential_convolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@Ѓ
uconvolutional_model_sequential_convolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@v
\convolutional_model_sequential_convolutional_block_3_conv2d_3_conv2d_readvariableop_resource:@@k
]convolutional_model_sequential_convolutional_block_3_conv2d_3_biasadd_readvariableop_resource:@p
bconvolutional_model_sequential_convolutional_block_3_batch_normalization_3_readvariableop_resource:@r
dconvolutional_model_sequential_convolutional_block_3_batch_normalization_3_readvariableop_1_resource:@Ђ
sconvolutional_model_sequential_convolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@Ѓ
uconvolutional_model_sequential_convolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@X
Econvolutional_model_sequential_1_dense_matmul_readvariableop_resource:	@ђU
Fconvolutional_model_sequential_1_dense_biasadd_readvariableop_resource:	ђZ
Gconvolutional_model_sequential_1_dense_1_matmul_readvariableop_resource:	ђV
Hconvolutional_model_sequential_1_dense_1_biasadd_readvariableop_resource:
identityѕбfconvolutional_model/sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpбhconvolutional_model/sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1бUconvolutional_model/sequential/convolutional_block/batch_normalization/ReadVariableOpбWconvolutional_model/sequential/convolutional_block/batch_normalization/ReadVariableOp_1бPconvolutional_model/sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOpбOconvolutional_model/sequential/convolutional_block/conv2d/Conv2D/ReadVariableOpбjconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpбlconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1бYconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/ReadVariableOpб[convolutional_model/sequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_1бTconvolutional_model/sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOpбSconvolutional_model/sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOpбjconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpбlconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1бYconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/ReadVariableOpб[convolutional_model/sequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_1бTconvolutional_model/sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOpбSconvolutional_model/sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOpбjconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpбlconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1бYconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/ReadVariableOpб[convolutional_model/sequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_1бTconvolutional_model/sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOpбSconvolutional_model/sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOpб=convolutional_model/sequential_1/dense/BiasAdd/ReadVariableOpб<convolutional_model/sequential_1/dense/MatMul/ReadVariableOpб?convolutional_model/sequential_1/dense_1/BiasAdd/ReadVariableOpб>convolutional_model/sequential_1/dense_1/MatMul/ReadVariableOp­
Oconvolutional_model/sequential/convolutional_block/conv2d/Conv2D/ReadVariableOpReadVariableOpXconvolutional_model_sequential_convolutional_block_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0Љ
@convolutional_model/sequential/convolutional_block/conv2d/Conv2DConv2Dinput_1Wconvolutional_model/sequential/convolutional_block/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ *
paddingVALID*
strides
Т
Pconvolutional_model/sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOpReadVariableOpYconvolutional_model_sequential_convolutional_block_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0Г
Aconvolutional_model/sequential/convolutional_block/conv2d/BiasAddBiasAddIconvolutional_model/sequential/convolutional_block/conv2d/Conv2D:output:0Xconvolutional_model/sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ ї
Hconvolutional_model/sequential/convolutional_block/max_pooling2d/MaxPoolMaxPoolJconvolutional_model/sequential/convolutional_block/conv2d/BiasAdd:output:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
­
Uconvolutional_model/sequential/convolutional_block/batch_normalization/ReadVariableOpReadVariableOp^convolutional_model_sequential_convolutional_block_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0З
Wconvolutional_model/sequential/convolutional_block/batch_normalization/ReadVariableOp_1ReadVariableOp`convolutional_model_sequential_convolutional_block_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0њ
fconvolutional_model/sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpoconvolutional_model_sequential_convolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ќ
hconvolutional_model/sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpqconvolutional_model_sequential_convolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0С
Wconvolutional_model/sequential/convolutional_block/batch_normalization/FusedBatchNormV3FusedBatchNormV3Qconvolutional_model/sequential/convolutional_block/max_pooling2d/MaxPool:output:0]convolutional_model/sequential/convolutional_block/batch_normalization/ReadVariableOp:value:0_convolutional_model/sequential/convolutional_block/batch_normalization/ReadVariableOp_1:value:0nconvolutional_model/sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0pconvolutional_model/sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( о
7convolutional_model/sequential/convolutional_block/ReluRelu[convolutional_model/sequential/convolutional_block/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          Э
Sconvolutional_model/sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp\convolutional_model_sequential_convolutional_block_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Н
Dconvolutional_model/sequential/convolutional_block_1/conv2d_1/Conv2DConv2DEconvolutional_model/sequential/convolutional_block/Relu:activations:0[convolutional_model/sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} *
paddingVALID*
strides
Ь
Tconvolutional_model/sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp]convolutional_model_sequential_convolutional_block_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0и
Econvolutional_model/sequential/convolutional_block_1/conv2d_1/BiasAddBiasAddMconvolutional_model/sequential/convolutional_block_1/conv2d_1/Conv2D:output:0\convolutional_model/sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} ћ
Lconvolutional_model/sequential/convolutional_block_1/max_pooling2d_1/MaxPoolMaxPoolNconvolutional_model/sequential/convolutional_block_1/conv2d_1/BiasAdd:output:0*/
_output_shapes
:         >> *
ksize
*
paddingVALID*
strides
Э
Yconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/ReadVariableOpReadVariableOpbconvolutional_model_sequential_convolutional_block_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype0Ч
[convolutional_model/sequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_1ReadVariableOpdconvolutional_model_sequential_convolutional_block_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype0џ
jconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpsconvolutional_model_sequential_convolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ъ
lconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpuconvolutional_model_sequential_convolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Ч
[convolutional_model/sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3Uconvolutional_model/sequential/convolutional_block_1/max_pooling2d_1/MaxPool:output:0aconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/ReadVariableOp:value:0cconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_1:value:0rconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0tconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         >> : : : : :*
epsilon%oЃ:*
is_training( ▄
9convolutional_model/sequential/convolutional_block_1/ReluRelu_convolutional_model/sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         >> Э
Sconvolutional_model/sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp\convolutional_model_sequential_convolutional_block_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0О
Dconvolutional_model/sequential/convolutional_block_2/conv2d_2/Conv2DConv2DGconvolutional_model/sequential/convolutional_block_1/Relu:activations:0[convolutional_model/sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@*
paddingVALID*
strides
Ь
Tconvolutional_model/sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp]convolutional_model_sequential_convolutional_block_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0и
Econvolutional_model/sequential/convolutional_block_2/conv2d_2/BiasAddBiasAddMconvolutional_model/sequential/convolutional_block_2/conv2d_2/Conv2D:output:0\convolutional_model/sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@ћ
Lconvolutional_model/sequential/convolutional_block_2/max_pooling2d_2/MaxPoolMaxPoolNconvolutional_model/sequential/convolutional_block_2/conv2d_2/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
Э
Yconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/ReadVariableOpReadVariableOpbconvolutional_model_sequential_convolutional_block_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0Ч
[convolutional_model/sequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_1ReadVariableOpdconvolutional_model_sequential_convolutional_block_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0џ
jconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpsconvolutional_model_sequential_convolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ъ
lconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpuconvolutional_model_sequential_convolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ч
[convolutional_model/sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3Uconvolutional_model/sequential/convolutional_block_2/max_pooling2d_2/MaxPool:output:0aconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/ReadVariableOp:value:0cconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_1:value:0rconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0tconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( ▄
9convolutional_model/sequential/convolutional_block_2/ReluRelu_convolutional_model/sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @Э
Sconvolutional_model/sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp\convolutional_model_sequential_convolutional_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0О
Dconvolutional_model/sequential/convolutional_block_3/conv2d_3/Conv2DConv2DGconvolutional_model/sequential/convolutional_block_2/Relu:activations:0[convolutional_model/sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
Ь
Tconvolutional_model/sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp]convolutional_model_sequential_convolutional_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0и
Econvolutional_model/sequential/convolutional_block_3/conv2d_3/BiasAddBiasAddMconvolutional_model/sequential/convolutional_block_3/conv2d_3/Conv2D:output:0\convolutional_model/sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ћ
Lconvolutional_model/sequential/convolutional_block_3/max_pooling2d_3/MaxPoolMaxPoolNconvolutional_model/sequential/convolutional_block_3/conv2d_3/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
Э
Yconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/ReadVariableOpReadVariableOpbconvolutional_model_sequential_convolutional_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0Ч
[convolutional_model/sequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpdconvolutional_model_sequential_convolutional_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0џ
jconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpsconvolutional_model_sequential_convolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ъ
lconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpuconvolutional_model_sequential_convolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ч
[convolutional_model/sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3Uconvolutional_model/sequential/convolutional_block_3/max_pooling2d_3/MaxPool:output:0aconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/ReadVariableOp:value:0cconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_1:value:0rconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0tconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( ▄
9convolutional_model/sequential/convolutional_block_3/ReluRelu_convolutional_model/sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @ћ
Cconvolutional_model/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      ѓ
1convolutional_model/global_average_pooling2d/MeanMeanGconvolutional_model/sequential/convolutional_block_3/Relu:activations:0Lconvolutional_model/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @├
<convolutional_model/sequential_1/dense/MatMul/ReadVariableOpReadVariableOpEconvolutional_model_sequential_1_dense_matmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0В
-convolutional_model/sequential_1/dense/MatMulMatMul:convolutional_model/global_average_pooling2d/Mean:output:0Dconvolutional_model/sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ┴
=convolutional_model/sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOpFconvolutional_model_sequential_1_dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0В
.convolutional_model/sequential_1/dense/BiasAddBiasAdd7convolutional_model/sequential_1/dense/MatMul:product:0Econvolutional_model/sequential_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЪ
+convolutional_model/sequential_1/dense/ReluRelu7convolutional_model/sequential_1/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђФ
1convolutional_model/sequential_1/dropout/IdentityIdentity9convolutional_model/sequential_1/dense/Relu:activations:0*
T0*(
_output_shapes
:         ђК
>convolutional_model/sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOpGconvolutional_model_sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0№
/convolutional_model/sequential_1/dense_1/MatMulMatMul:convolutional_model/sequential_1/dropout/Identity:output:0Fconvolutional_model/sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ─
?convolutional_model/sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOpHconvolutional_model_sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ы
0convolutional_model/sequential_1/dense_1/BiasAddBiasAdd9convolutional_model/sequential_1/dense_1/MatMul:product:0Gconvolutional_model/sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         е
0convolutional_model/sequential_1/dense_1/SoftmaxSoftmax9convolutional_model/sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         Ѕ
IdentityIdentity:convolutional_model/sequential_1/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ╝
NoOpNoOpg^convolutional_model/sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpi^convolutional_model/sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1V^convolutional_model/sequential/convolutional_block/batch_normalization/ReadVariableOpX^convolutional_model/sequential/convolutional_block/batch_normalization/ReadVariableOp_1Q^convolutional_model/sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOpP^convolutional_model/sequential/convolutional_block/conv2d/Conv2D/ReadVariableOpk^convolutional_model/sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpm^convolutional_model/sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Z^convolutional_model/sequential/convolutional_block_1/batch_normalization_1/ReadVariableOp\^convolutional_model/sequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_1U^convolutional_model/sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOpT^convolutional_model/sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOpk^convolutional_model/sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpm^convolutional_model/sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Z^convolutional_model/sequential/convolutional_block_2/batch_normalization_2/ReadVariableOp\^convolutional_model/sequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_1U^convolutional_model/sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOpT^convolutional_model/sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOpk^convolutional_model/sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpm^convolutional_model/sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Z^convolutional_model/sequential/convolutional_block_3/batch_normalization_3/ReadVariableOp\^convolutional_model/sequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_1U^convolutional_model/sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOpT^convolutional_model/sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp>^convolutional_model/sequential_1/dense/BiasAdd/ReadVariableOp=^convolutional_model/sequential_1/dense/MatMul/ReadVariableOp@^convolutional_model/sequential_1/dense_1/BiasAdd/ReadVariableOp?^convolutional_model/sequential_1/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2л
fconvolutional_model/sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpfconvolutional_model/sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp2н
hconvolutional_model/sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1hconvolutional_model/sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_12«
Uconvolutional_model/sequential/convolutional_block/batch_normalization/ReadVariableOpUconvolutional_model/sequential/convolutional_block/batch_normalization/ReadVariableOp2▓
Wconvolutional_model/sequential/convolutional_block/batch_normalization/ReadVariableOp_1Wconvolutional_model/sequential/convolutional_block/batch_normalization/ReadVariableOp_12ц
Pconvolutional_model/sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOpPconvolutional_model/sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOp2б
Oconvolutional_model/sequential/convolutional_block/conv2d/Conv2D/ReadVariableOpOconvolutional_model/sequential/convolutional_block/conv2d/Conv2D/ReadVariableOp2п
jconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpjconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2▄
lconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1lconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12Х
Yconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/ReadVariableOpYconvolutional_model/sequential/convolutional_block_1/batch_normalization_1/ReadVariableOp2║
[convolutional_model/sequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_1[convolutional_model/sequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_12г
Tconvolutional_model/sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOpTconvolutional_model/sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp2ф
Sconvolutional_model/sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOpSconvolutional_model/sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOp2п
jconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpjconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2▄
lconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1lconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12Х
Yconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/ReadVariableOpYconvolutional_model/sequential/convolutional_block_2/batch_normalization_2/ReadVariableOp2║
[convolutional_model/sequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_1[convolutional_model/sequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_12г
Tconvolutional_model/sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOpTconvolutional_model/sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp2ф
Sconvolutional_model/sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOpSconvolutional_model/sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOp2п
jconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpjconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2▄
lconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1lconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12Х
Yconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/ReadVariableOpYconvolutional_model/sequential/convolutional_block_3/batch_normalization_3/ReadVariableOp2║
[convolutional_model/sequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_1[convolutional_model/sequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_12г
Tconvolutional_model/sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOpTconvolutional_model/sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp2ф
Sconvolutional_model/sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOpSconvolutional_model/sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp2~
=convolutional_model/sequential_1/dense/BiasAdd/ReadVariableOp=convolutional_model/sequential_1/dense/BiasAdd/ReadVariableOp2|
<convolutional_model/sequential_1/dense/MatMul/ReadVariableOp<convolutional_model/sequential_1/dense/MatMul/ReadVariableOp2ѓ
?convolutional_model/sequential_1/dense_1/BiasAdd/ReadVariableOp?convolutional_model/sequential_1/dense_1/BiasAdd/ReadVariableOp2ђ
>convolutional_model/sequential_1/dense_1/MatMul/ReadVariableOp>convolutional_model/sequential_1/dense_1/MatMul/ReadVariableOp:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1
х
I
-__inference_max_pooling2d_layer_call_fn_67939

inputs
identity┘
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_64823Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Д

Ч
C__inference_conv2d_2_layer_call_and_return_conditional_losses_68116

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         <<@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         >> : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         >> 
 
_user_specified_nameinputs
Ц	
ј
5__inference_convolutional_block_2_layer_call_fn_67693

inputs!
unknown: @
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_65345w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         >> : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         >> 
 
_user_specified_nameinputs
Љ%
╗
N__inference_convolutional_block_layer_call_and_return_conditional_losses_67590

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: 
identityѕб"batch_normalization/AssignNewValueб$batch_normalization/AssignNewValue_1б3batch_normalization/FusedBatchNormV3/ReadVariableOpб5batch_normalization/FusedBatchNormV3/ReadVariableOp_1б"batch_normalization/ReadVariableOpб$batch_normalization/ReadVariableOp_1бconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpі
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ф
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ *
paddingVALID*
strides
ђ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ћ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ д
max_pooling2d/MaxPoolMaxPoolconv2d/BiasAdd:output:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
і
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0ј
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0г
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0░
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0└
$batch_normalization/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<ќ
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(а
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(p
ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          І
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
┌
Ѕ
*__inference_sequential_layer_call_fn_65691
convolutional_block_input!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityѕбStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallconvolutional_block_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_65640w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:l h
1
_output_shapes
:         ђђ
3
_user_specified_nameconvolutional_block_input
Ё
┐
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_65126

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Џ
с
#__inference_signature_wrapper_66785
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	@ђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *)
f$R"
 __inference__wrapped_model_64814o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1
њ	
л
5__inference_batch_normalization_3_layer_call_fn_68230

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_65441Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Џ&
┘
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_67848

inputsA
'conv2d_3_conv2d_readvariableop_resource:@@6
(conv2d_3_biasadd_readvariableop_resource:@;
-batch_normalization_3_readvariableop_resource:@=
/batch_normalization_3_readvariableop_1_resource:@L
>batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб$batch_normalization_3/AssignNewValueб&batch_normalization_3/AssignNewValue_1б5batch_normalization_3/FusedBatchNormV3/ReadVariableOpб7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_3/ReadVariableOpб&batch_normalization_3/ReadVariableOp_1бconv2d_3/BiasAdd/ReadVariableOpбconv2d_3/Conv2D/ReadVariableOpј
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0г
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
ё
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ў
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @ф
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
ј
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0њ
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╠
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_3/MaxPool:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ъ
$batch_normalization_3/AssignNewValueAssignVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(е
&batch_normalization_3/AssignNewValue_1AssignVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(r
ReluRelu*batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @Џ
NoOpNoOp%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @: : : : : : 2L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Д

Ч
C__inference_conv2d_1_layer_call_and_return_conditional_losses_68025

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         }} w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Л┬
ю%
N__inference_convolutional_model_layer_call_and_return_conditional_losses_67016

inputs^
Dsequential_convolutional_block_conv2d_conv2d_readvariableop_resource: S
Esequential_convolutional_block_conv2d_biasadd_readvariableop_resource: X
Jsequential_convolutional_block_batch_normalization_readvariableop_resource: Z
Lsequential_convolutional_block_batch_normalization_readvariableop_1_resource: i
[sequential_convolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_resource: k
]sequential_convolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: b
Hsequential_convolutional_block_1_conv2d_1_conv2d_readvariableop_resource:  W
Isequential_convolutional_block_1_conv2d_1_biasadd_readvariableop_resource: \
Nsequential_convolutional_block_1_batch_normalization_1_readvariableop_resource: ^
Psequential_convolutional_block_1_batch_normalization_1_readvariableop_1_resource: m
_sequential_convolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource: o
asequential_convolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: b
Hsequential_convolutional_block_2_conv2d_2_conv2d_readvariableop_resource: @W
Isequential_convolutional_block_2_conv2d_2_biasadd_readvariableop_resource:@\
Nsequential_convolutional_block_2_batch_normalization_2_readvariableop_resource:@^
Psequential_convolutional_block_2_batch_normalization_2_readvariableop_1_resource:@m
_sequential_convolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@o
asequential_convolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@b
Hsequential_convolutional_block_3_conv2d_3_conv2d_readvariableop_resource:@@W
Isequential_convolutional_block_3_conv2d_3_biasadd_readvariableop_resource:@\
Nsequential_convolutional_block_3_batch_normalization_3_readvariableop_resource:@^
Psequential_convolutional_block_3_batch_normalization_3_readvariableop_1_resource:@m
_sequential_convolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@o
asequential_convolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@D
1sequential_1_dense_matmul_readvariableop_resource:	@ђA
2sequential_1_dense_biasadd_readvariableop_resource:	ђF
3sequential_1_dense_1_matmul_readvariableop_resource:	ђB
4sequential_1_dense_1_biasadd_readvariableop_resource:
identityѕбRsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpбTsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1бAsequential/convolutional_block/batch_normalization/ReadVariableOpбCsequential/convolutional_block/batch_normalization/ReadVariableOp_1б<sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOpб;sequential/convolutional_block/conv2d/Conv2D/ReadVariableOpбVsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpбXsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1бEsequential/convolutional_block_1/batch_normalization_1/ReadVariableOpбGsequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_1б@sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOpб?sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOpбVsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpбXsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1бEsequential/convolutional_block_2/batch_normalization_2/ReadVariableOpбGsequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_1б@sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOpб?sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOpбVsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpбXsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1бEsequential/convolutional_block_3/batch_normalization_3/ReadVariableOpбGsequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_1б@sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOpб?sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOpб)sequential_1/dense/BiasAdd/ReadVariableOpб(sequential_1/dense/MatMul/ReadVariableOpб+sequential_1/dense_1/BiasAdd/ReadVariableOpб*sequential_1/dense_1/MatMul/ReadVariableOp╚
;sequential/convolutional_block/conv2d/Conv2D/ReadVariableOpReadVariableOpDsequential_convolutional_block_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0У
,sequential/convolutional_block/conv2d/Conv2DConv2DinputsCsequential/convolutional_block/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ *
paddingVALID*
strides
Й
<sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOpReadVariableOpEsequential_convolutional_block_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ы
-sequential/convolutional_block/conv2d/BiasAddBiasAdd5sequential/convolutional_block/conv2d/Conv2D:output:0Dsequential/convolutional_block/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ С
4sequential/convolutional_block/max_pooling2d/MaxPoolMaxPool6sequential/convolutional_block/conv2d/BiasAdd:output:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
╚
Asequential/convolutional_block/batch_normalization/ReadVariableOpReadVariableOpJsequential_convolutional_block_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0╠
Csequential/convolutional_block/batch_normalization/ReadVariableOp_1ReadVariableOpLsequential_convolutional_block_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0Ж
Rsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp[sequential_convolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ь
Tsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]sequential_convolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0В
Csequential/convolutional_block/batch_normalization/FusedBatchNormV3FusedBatchNormV3=sequential/convolutional_block/max_pooling2d/MaxPool:output:0Isequential/convolutional_block/batch_normalization/ReadVariableOp:value:0Ksequential/convolutional_block/batch_normalization/ReadVariableOp_1:value:0Zsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0\sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( «
#sequential/convolutional_block/ReluReluGsequential/convolutional_block/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          л
?sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpHsequential_convolutional_block_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Ў
0sequential/convolutional_block_1/conv2d_1/Conv2DConv2D1sequential/convolutional_block/Relu:activations:0Gsequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} *
paddingVALID*
strides
к
@sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpIsequential_convolutional_block_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ч
1sequential/convolutional_block_1/conv2d_1/BiasAddBiasAdd9sequential/convolutional_block_1/conv2d_1/Conv2D:output:0Hsequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} В
8sequential/convolutional_block_1/max_pooling2d_1/MaxPoolMaxPool:sequential/convolutional_block_1/conv2d_1/BiasAdd:output:0*/
_output_shapes
:         >> *
ksize
*
paddingVALID*
strides
л
Esequential/convolutional_block_1/batch_normalization_1/ReadVariableOpReadVariableOpNsequential_convolutional_block_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype0н
Gsequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_1ReadVariableOpPsequential_convolutional_block_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype0Ы
Vsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp_sequential_convolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Ш
Xsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpasequential_convolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ё
Gsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3Asequential/convolutional_block_1/max_pooling2d_1/MaxPool:output:0Msequential/convolutional_block_1/batch_normalization_1/ReadVariableOp:value:0Osequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_1:value:0^sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0`sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         >> : : : : :*
epsilon%oЃ:*
is_training( ┤
%sequential/convolutional_block_1/ReluReluKsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         >> л
?sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOpHsequential_convolutional_block_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Џ
0sequential/convolutional_block_2/conv2d_2/Conv2DConv2D3sequential/convolutional_block_1/Relu:activations:0Gsequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@*
paddingVALID*
strides
к
@sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpIsequential_convolutional_block_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ч
1sequential/convolutional_block_2/conv2d_2/BiasAddBiasAdd9sequential/convolutional_block_2/conv2d_2/Conv2D:output:0Hsequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@В
8sequential/convolutional_block_2/max_pooling2d_2/MaxPoolMaxPool:sequential/convolutional_block_2/conv2d_2/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
л
Esequential/convolutional_block_2/batch_normalization_2/ReadVariableOpReadVariableOpNsequential_convolutional_block_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0н
Gsequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_1ReadVariableOpPsequential_convolutional_block_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ы
Vsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp_sequential_convolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
Xsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpasequential_convolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ё
Gsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3Asequential/convolutional_block_2/max_pooling2d_2/MaxPool:output:0Msequential/convolutional_block_2/batch_normalization_2/ReadVariableOp:value:0Osequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_1:value:0^sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0`sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( ┤
%sequential/convolutional_block_2/ReluReluKsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @л
?sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpHsequential_convolutional_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Џ
0sequential/convolutional_block_3/conv2d_3/Conv2DConv2D3sequential/convolutional_block_2/Relu:activations:0Gsequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
к
@sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpIsequential_convolutional_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ч
1sequential/convolutional_block_3/conv2d_3/BiasAddBiasAdd9sequential/convolutional_block_3/conv2d_3/Conv2D:output:0Hsequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @В
8sequential/convolutional_block_3/max_pooling2d_3/MaxPoolMaxPool:sequential/convolutional_block_3/conv2d_3/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
л
Esequential/convolutional_block_3/batch_normalization_3/ReadVariableOpReadVariableOpNsequential_convolutional_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0н
Gsequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpPsequential_convolutional_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ы
Vsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp_sequential_convolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ш
Xsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpasequential_convolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0ё
Gsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3Asequential/convolutional_block_3/max_pooling2d_3/MaxPool:output:0Msequential/convolutional_block_3/batch_normalization_3/ReadVariableOp:value:0Osequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_1:value:0^sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0`sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( ┤
%sequential/convolutional_block_3/ReluReluKsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @ђ
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      к
global_average_pooling2d/MeanMean3sequential/convolutional_block_3/Relu:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*'
_output_shapes
:         @Џ
(sequential_1/dense/MatMul/ReadVariableOpReadVariableOp1sequential_1_dense_matmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0░
sequential_1/dense/MatMulMatMul&global_average_pooling2d/Mean:output:00sequential_1/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЎ
)sequential_1/dense/BiasAdd/ReadVariableOpReadVariableOp2sequential_1_dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0░
sequential_1/dense/BiasAddBiasAdd#sequential_1/dense/MatMul:product:01sequential_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђw
sequential_1/dense/ReluRelu#sequential_1/dense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђЃ
sequential_1/dropout/IdentityIdentity%sequential_1/dense/Relu:activations:0*
T0*(
_output_shapes
:         ђЪ
*sequential_1/dense_1/MatMul/ReadVariableOpReadVariableOp3sequential_1_dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0│
sequential_1/dense_1/MatMulMatMul&sequential_1/dropout/Identity:output:02sequential_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ю
+sequential_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp4sequential_1_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0х
sequential_1/dense_1/BiasAddBiasAdd%sequential_1/dense_1/MatMul:product:03sequential_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ђ
sequential_1/dense_1/SoftmaxSoftmax%sequential_1/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         u
IdentityIdentity&sequential_1/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ї
NoOpNoOpS^sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpU^sequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1B^sequential/convolutional_block/batch_normalization/ReadVariableOpD^sequential/convolutional_block/batch_normalization/ReadVariableOp_1=^sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOp<^sequential/convolutional_block/conv2d/Conv2D/ReadVariableOpW^sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpY^sequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1F^sequential/convolutional_block_1/batch_normalization_1/ReadVariableOpH^sequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_1A^sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp@^sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOpW^sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpY^sequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1F^sequential/convolutional_block_2/batch_normalization_2/ReadVariableOpH^sequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_1A^sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp@^sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOpW^sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpY^sequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1F^sequential/convolutional_block_3/batch_normalization_3/ReadVariableOpH^sequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_1A^sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp@^sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp*^sequential_1/dense/BiasAdd/ReadVariableOp)^sequential_1/dense/MatMul/ReadVariableOp,^sequential_1/dense_1/BiasAdd/ReadVariableOp+^sequential_1/dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2е
Rsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpRsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp2г
Tsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Tsequential/convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_12є
Asequential/convolutional_block/batch_normalization/ReadVariableOpAsequential/convolutional_block/batch_normalization/ReadVariableOp2і
Csequential/convolutional_block/batch_normalization/ReadVariableOp_1Csequential/convolutional_block/batch_normalization/ReadVariableOp_12|
<sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOp<sequential/convolutional_block/conv2d/BiasAdd/ReadVariableOp2z
;sequential/convolutional_block/conv2d/Conv2D/ReadVariableOp;sequential/convolutional_block/conv2d/Conv2D/ReadVariableOp2░
Vsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpVsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2┤
Xsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Xsequential/convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12ј
Esequential/convolutional_block_1/batch_normalization_1/ReadVariableOpEsequential/convolutional_block_1/batch_normalization_1/ReadVariableOp2њ
Gsequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_1Gsequential/convolutional_block_1/batch_normalization_1/ReadVariableOp_12ё
@sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp@sequential/convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp2ѓ
?sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOp?sequential/convolutional_block_1/conv2d_1/Conv2D/ReadVariableOp2░
Vsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpVsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2┤
Xsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Xsequential/convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12ј
Esequential/convolutional_block_2/batch_normalization_2/ReadVariableOpEsequential/convolutional_block_2/batch_normalization_2/ReadVariableOp2њ
Gsequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_1Gsequential/convolutional_block_2/batch_normalization_2/ReadVariableOp_12ё
@sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp@sequential/convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp2ѓ
?sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOp?sequential/convolutional_block_2/conv2d_2/Conv2D/ReadVariableOp2░
Vsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpVsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2┤
Xsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Xsequential/convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12ј
Esequential/convolutional_block_3/batch_normalization_3/ReadVariableOpEsequential/convolutional_block_3/batch_normalization_3/ReadVariableOp2њ
Gsequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_1Gsequential/convolutional_block_3/batch_normalization_3/ReadVariableOp_12ё
@sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp@sequential/convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp2ѓ
?sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp?sequential/convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp2V
)sequential_1/dense/BiasAdd/ReadVariableOp)sequential_1/dense/BiasAdd/ReadVariableOp2T
(sequential_1/dense/MatMul/ReadVariableOp(sequential_1/dense/MatMul/ReadVariableOp2Z
+sequential_1/dense_1/BiasAdd/ReadVariableOp+sequential_1/dense_1/BiasAdd/ReadVariableOp2X
*sequential_1/dense_1/MatMul/ReadVariableOp*sequential_1/dense_1/MatMul/ReadVariableOp:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
─
Х
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_65345

inputs(
conv2d_2_65328: @
conv2d_2_65330:@)
batch_normalization_2_65334:@)
batch_normalization_2_65336:@)
batch_normalization_2_65338:@)
batch_normalization_2_65340:@
identityѕб-batch_normalization_2/StatefulPartitionedCallб conv2d_2/StatefulPartitionedCallЭ
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_65328conv2d_2_65330*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         <<@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_65327з
max_pooling2d_2/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_65243ї
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_2/PartitionedCall:output:0batch_normalization_2_65334batch_normalization_2_65336batch_normalization_2_65338batch_normalization_2_65340*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_65268~
ReluRelu6batch_normalization_2/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @Ў
NoOpNoOp.^batch_normalization_2/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         >> : : : : : : 2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall:W S
/
_output_shapes
:         >> 
 
_user_specified_nameinputs
Б	
ї
3__inference_convolutional_block_layer_call_fn_67538

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identityѕбStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_convolutional_block_layer_call_and_return_conditional_losses_64989w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
§й
ц!
E__inference_sequential_layer_call_and_return_conditional_losses_67422

inputsS
9convolutional_block_conv2d_conv2d_readvariableop_resource: H
:convolutional_block_conv2d_biasadd_readvariableop_resource: M
?convolutional_block_batch_normalization_readvariableop_resource: O
Aconvolutional_block_batch_normalization_readvariableop_1_resource: ^
Pconvolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_resource: `
Rconvolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: W
=convolutional_block_1_conv2d_1_conv2d_readvariableop_resource:  L
>convolutional_block_1_conv2d_1_biasadd_readvariableop_resource: Q
Cconvolutional_block_1_batch_normalization_1_readvariableop_resource: S
Econvolutional_block_1_batch_normalization_1_readvariableop_1_resource: b
Tconvolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource: d
Vconvolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: W
=convolutional_block_2_conv2d_2_conv2d_readvariableop_resource: @L
>convolutional_block_2_conv2d_2_biasadd_readvariableop_resource:@Q
Cconvolutional_block_2_batch_normalization_2_readvariableop_resource:@S
Econvolutional_block_2_batch_normalization_2_readvariableop_1_resource:@b
Tconvolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@d
Vconvolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@W
=convolutional_block_3_conv2d_3_conv2d_readvariableop_resource:@@L
>convolutional_block_3_conv2d_3_biasadd_readvariableop_resource:@Q
Cconvolutional_block_3_batch_normalization_3_readvariableop_resource:@S
Econvolutional_block_3_batch_normalization_3_readvariableop_1_resource:@b
Tconvolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@d
Vconvolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб6convolutional_block/batch_normalization/AssignNewValueб8convolutional_block/batch_normalization/AssignNewValue_1бGconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpбIconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1б6convolutional_block/batch_normalization/ReadVariableOpб8convolutional_block/batch_normalization/ReadVariableOp_1б1convolutional_block/conv2d/BiasAdd/ReadVariableOpб0convolutional_block/conv2d/Conv2D/ReadVariableOpб:convolutional_block_1/batch_normalization_1/AssignNewValueб<convolutional_block_1/batch_normalization_1/AssignNewValue_1бKconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpбMconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б:convolutional_block_1/batch_normalization_1/ReadVariableOpб<convolutional_block_1/batch_normalization_1/ReadVariableOp_1б5convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOpб4convolutional_block_1/conv2d_1/Conv2D/ReadVariableOpб:convolutional_block_2/batch_normalization_2/AssignNewValueб<convolutional_block_2/batch_normalization_2/AssignNewValue_1бKconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpбMconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б:convolutional_block_2/batch_normalization_2/ReadVariableOpб<convolutional_block_2/batch_normalization_2/ReadVariableOp_1б5convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOpб4convolutional_block_2/conv2d_2/Conv2D/ReadVariableOpб:convolutional_block_3/batch_normalization_3/AssignNewValueб<convolutional_block_3/batch_normalization_3/AssignNewValue_1бKconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpбMconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б:convolutional_block_3/batch_normalization_3/ReadVariableOpб<convolutional_block_3/batch_normalization_3/ReadVariableOp_1б5convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOpб4convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp▓
0convolutional_block/conv2d/Conv2D/ReadVariableOpReadVariableOp9convolutional_block_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0м
!convolutional_block/conv2d/Conv2DConv2Dinputs8convolutional_block/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ *
paddingVALID*
strides
е
1convolutional_block/conv2d/BiasAdd/ReadVariableOpReadVariableOp:convolutional_block_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0л
"convolutional_block/conv2d/BiasAddBiasAdd*convolutional_block/conv2d/Conv2D:output:09convolutional_block/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ ╬
)convolutional_block/max_pooling2d/MaxPoolMaxPool+convolutional_block/conv2d/BiasAdd:output:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
▓
6convolutional_block/batch_normalization/ReadVariableOpReadVariableOp?convolutional_block_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0Х
8convolutional_block/batch_normalization/ReadVariableOp_1ReadVariableOpAconvolutional_block_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0н
Gconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpPconvolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0п
Iconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRconvolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0И
8convolutional_block/batch_normalization/FusedBatchNormV3FusedBatchNormV32convolutional_block/max_pooling2d/MaxPool:output:0>convolutional_block/batch_normalization/ReadVariableOp:value:0@convolutional_block/batch_normalization/ReadVariableOp_1:value:0Oconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Qconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<Т
6convolutional_block/batch_normalization/AssignNewValueAssignVariableOpPconvolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_resourceEconvolutional_block/batch_normalization/FusedBatchNormV3:batch_mean:0H^convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(­
8convolutional_block/batch_normalization/AssignNewValue_1AssignVariableOpRconvolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceIconvolutional_block/batch_normalization/FusedBatchNormV3:batch_variance:0J^convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(ў
convolutional_block/ReluRelu<convolutional_block/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          ║
4convolutional_block_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp=convolutional_block_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Э
%convolutional_block_1/conv2d_1/Conv2DConv2D&convolutional_block/Relu:activations:0<convolutional_block_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} *
paddingVALID*
strides
░
5convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp>convolutional_block_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┌
&convolutional_block_1/conv2d_1/BiasAddBiasAdd.convolutional_block_1/conv2d_1/Conv2D:output:0=convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} о
-convolutional_block_1/max_pooling2d_1/MaxPoolMaxPool/convolutional_block_1/conv2d_1/BiasAdd:output:0*/
_output_shapes
:         >> *
ksize
*
paddingVALID*
strides
║
:convolutional_block_1/batch_normalization_1/ReadVariableOpReadVariableOpCconvolutional_block_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype0Й
<convolutional_block_1/batch_normalization_1/ReadVariableOp_1ReadVariableOpEconvolutional_block_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype0▄
Kconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpTconvolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Я
Mconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVconvolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0л
<convolutional_block_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV36convolutional_block_1/max_pooling2d_1/MaxPool:output:0Bconvolutional_block_1/batch_normalization_1/ReadVariableOp:value:0Dconvolutional_block_1/batch_normalization_1/ReadVariableOp_1:value:0Sconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Uconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         >> : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<Ш
:convolutional_block_1/batch_normalization_1/AssignNewValueAssignVariableOpTconvolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceIconvolutional_block_1/batch_normalization_1/FusedBatchNormV3:batch_mean:0L^convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ђ
<convolutional_block_1/batch_normalization_1/AssignNewValue_1AssignVariableOpVconvolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resourceMconvolutional_block_1/batch_normalization_1/FusedBatchNormV3:batch_variance:0N^convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(ъ
convolutional_block_1/ReluRelu@convolutional_block_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         >> ║
4convolutional_block_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp=convolutional_block_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
%convolutional_block_2/conv2d_2/Conv2DConv2D(convolutional_block_1/Relu:activations:0<convolutional_block_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@*
paddingVALID*
strides
░
5convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp>convolutional_block_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┌
&convolutional_block_2/conv2d_2/BiasAddBiasAdd.convolutional_block_2/conv2d_2/Conv2D:output:0=convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@о
-convolutional_block_2/max_pooling2d_2/MaxPoolMaxPool/convolutional_block_2/conv2d_2/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
║
:convolutional_block_2/batch_normalization_2/ReadVariableOpReadVariableOpCconvolutional_block_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0Й
<convolutional_block_2/batch_normalization_2/ReadVariableOp_1ReadVariableOpEconvolutional_block_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0▄
Kconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpTconvolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Я
Mconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVconvolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0л
<convolutional_block_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV36convolutional_block_2/max_pooling2d_2/MaxPool:output:0Bconvolutional_block_2/batch_normalization_2/ReadVariableOp:value:0Dconvolutional_block_2/batch_normalization_2/ReadVariableOp_1:value:0Sconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Uconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<Ш
:convolutional_block_2/batch_normalization_2/AssignNewValueAssignVariableOpTconvolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceIconvolutional_block_2/batch_normalization_2/FusedBatchNormV3:batch_mean:0L^convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ђ
<convolutional_block_2/batch_normalization_2/AssignNewValue_1AssignVariableOpVconvolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceMconvolutional_block_2/batch_normalization_2/FusedBatchNormV3:batch_variance:0N^convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(ъ
convolutional_block_2/ReluRelu@convolutional_block_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @║
4convolutional_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp=convolutional_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Щ
%convolutional_block_3/conv2d_3/Conv2DConv2D(convolutional_block_2/Relu:activations:0<convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
░
5convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp>convolutional_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┌
&convolutional_block_3/conv2d_3/BiasAddBiasAdd.convolutional_block_3/conv2d_3/Conv2D:output:0=convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @о
-convolutional_block_3/max_pooling2d_3/MaxPoolMaxPool/convolutional_block_3/conv2d_3/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
║
:convolutional_block_3/batch_normalization_3/ReadVariableOpReadVariableOpCconvolutional_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0Й
<convolutional_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpEconvolutional_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0▄
Kconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpTconvolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Я
Mconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVconvolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0л
<convolutional_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV36convolutional_block_3/max_pooling2d_3/MaxPool:output:0Bconvolutional_block_3/batch_normalization_3/ReadVariableOp:value:0Dconvolutional_block_3/batch_normalization_3/ReadVariableOp_1:value:0Sconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Uconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<Ш
:convolutional_block_3/batch_normalization_3/AssignNewValueAssignVariableOpTconvolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceIconvolutional_block_3/batch_normalization_3/FusedBatchNormV3:batch_mean:0L^convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(ђ
<convolutional_block_3/batch_normalization_3/AssignNewValue_1AssignVariableOpVconvolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceMconvolutional_block_3/batch_normalization_3/FusedBatchNormV3:batch_variance:0N^convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(ъ
convolutional_block_3/ReluRelu@convolutional_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @
IdentityIdentity(convolutional_block_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:         @║
NoOpNoOp7^convolutional_block/batch_normalization/AssignNewValue9^convolutional_block/batch_normalization/AssignNewValue_1H^convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpJ^convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_17^convolutional_block/batch_normalization/ReadVariableOp9^convolutional_block/batch_normalization/ReadVariableOp_12^convolutional_block/conv2d/BiasAdd/ReadVariableOp1^convolutional_block/conv2d/Conv2D/ReadVariableOp;^convolutional_block_1/batch_normalization_1/AssignNewValue=^convolutional_block_1/batch_normalization_1/AssignNewValue_1L^convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpN^convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1;^convolutional_block_1/batch_normalization_1/ReadVariableOp=^convolutional_block_1/batch_normalization_1/ReadVariableOp_16^convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp5^convolutional_block_1/conv2d_1/Conv2D/ReadVariableOp;^convolutional_block_2/batch_normalization_2/AssignNewValue=^convolutional_block_2/batch_normalization_2/AssignNewValue_1L^convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpN^convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1;^convolutional_block_2/batch_normalization_2/ReadVariableOp=^convolutional_block_2/batch_normalization_2/ReadVariableOp_16^convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp5^convolutional_block_2/conv2d_2/Conv2D/ReadVariableOp;^convolutional_block_3/batch_normalization_3/AssignNewValue=^convolutional_block_3/batch_normalization_3/AssignNewValue_1L^convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpN^convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1;^convolutional_block_3/batch_normalization_3/ReadVariableOp=^convolutional_block_3/batch_normalization_3/ReadVariableOp_16^convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp5^convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : 2p
6convolutional_block/batch_normalization/AssignNewValue6convolutional_block/batch_normalization/AssignNewValue2t
8convolutional_block/batch_normalization/AssignNewValue_18convolutional_block/batch_normalization/AssignNewValue_12њ
Gconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpGconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp2ќ
Iconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Iconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_12p
6convolutional_block/batch_normalization/ReadVariableOp6convolutional_block/batch_normalization/ReadVariableOp2t
8convolutional_block/batch_normalization/ReadVariableOp_18convolutional_block/batch_normalization/ReadVariableOp_12f
1convolutional_block/conv2d/BiasAdd/ReadVariableOp1convolutional_block/conv2d/BiasAdd/ReadVariableOp2d
0convolutional_block/conv2d/Conv2D/ReadVariableOp0convolutional_block/conv2d/Conv2D/ReadVariableOp2x
:convolutional_block_1/batch_normalization_1/AssignNewValue:convolutional_block_1/batch_normalization_1/AssignNewValue2|
<convolutional_block_1/batch_normalization_1/AssignNewValue_1<convolutional_block_1/batch_normalization_1/AssignNewValue_12џ
Kconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpKconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2ъ
Mconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Mconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12x
:convolutional_block_1/batch_normalization_1/ReadVariableOp:convolutional_block_1/batch_normalization_1/ReadVariableOp2|
<convolutional_block_1/batch_normalization_1/ReadVariableOp_1<convolutional_block_1/batch_normalization_1/ReadVariableOp_12n
5convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp5convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp2l
4convolutional_block_1/conv2d_1/Conv2D/ReadVariableOp4convolutional_block_1/conv2d_1/Conv2D/ReadVariableOp2x
:convolutional_block_2/batch_normalization_2/AssignNewValue:convolutional_block_2/batch_normalization_2/AssignNewValue2|
<convolutional_block_2/batch_normalization_2/AssignNewValue_1<convolutional_block_2/batch_normalization_2/AssignNewValue_12џ
Kconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpKconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2ъ
Mconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Mconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12x
:convolutional_block_2/batch_normalization_2/ReadVariableOp:convolutional_block_2/batch_normalization_2/ReadVariableOp2|
<convolutional_block_2/batch_normalization_2/ReadVariableOp_1<convolutional_block_2/batch_normalization_2/ReadVariableOp_12n
5convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp5convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp2l
4convolutional_block_2/conv2d_2/Conv2D/ReadVariableOp4convolutional_block_2/conv2d_2/Conv2D/ReadVariableOp2x
:convolutional_block_3/batch_normalization_3/AssignNewValue:convolutional_block_3/batch_normalization_3/AssignNewValue2|
<convolutional_block_3/batch_normalization_3/AssignNewValue_1<convolutional_block_3/batch_normalization_3/AssignNewValue_12џ
Kconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpKconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2ъ
Mconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Mconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12x
:convolutional_block_3/batch_normalization_3/ReadVariableOp:convolutional_block_3/batch_normalization_3/ReadVariableOp2|
<convolutional_block_3/batch_normalization_3/ReadVariableOp_1<convolutional_block_3/batch_normalization_3/ReadVariableOp_12n
5convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp5convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp2l
4convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp4convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
▄
┼
G__inference_sequential_1_layer_call_and_return_conditional_losses_67504

inputs7
$dense_matmul_readvariableop_resource:	@ђ4
%dense_biasadd_readvariableop_resource:	ђ9
&dense_1_matmul_readvariableop_resource:	ђ5
'dense_1_biasadd_readvariableop_resource:
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбdense_1/BiasAdd/ReadVariableOpбdense_1/MatMul/ReadVariableOpЂ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	@ђ*
dtype0v
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:ђ*
dtype0Ѕ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:         ђZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?Є
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:         ђ]
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:Е
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*

seedc
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>┐
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђђ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђѓ
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:         ђЁ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	ђ*
dtype0ї
dense_1/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ѓ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0ј
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:         ─
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
┘
`
B__inference_dropout_layer_call_and_return_conditional_losses_67883

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
Нќ
╝
E__inference_sequential_layer_call_and_return_conditional_losses_67330

inputsS
9convolutional_block_conv2d_conv2d_readvariableop_resource: H
:convolutional_block_conv2d_biasadd_readvariableop_resource: M
?convolutional_block_batch_normalization_readvariableop_resource: O
Aconvolutional_block_batch_normalization_readvariableop_1_resource: ^
Pconvolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_resource: `
Rconvolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: W
=convolutional_block_1_conv2d_1_conv2d_readvariableop_resource:  L
>convolutional_block_1_conv2d_1_biasadd_readvariableop_resource: Q
Cconvolutional_block_1_batch_normalization_1_readvariableop_resource: S
Econvolutional_block_1_batch_normalization_1_readvariableop_1_resource: b
Tconvolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource: d
Vconvolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: W
=convolutional_block_2_conv2d_2_conv2d_readvariableop_resource: @L
>convolutional_block_2_conv2d_2_biasadd_readvariableop_resource:@Q
Cconvolutional_block_2_batch_normalization_2_readvariableop_resource:@S
Econvolutional_block_2_batch_normalization_2_readvariableop_1_resource:@b
Tconvolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@d
Vconvolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@W
=convolutional_block_3_conv2d_3_conv2d_readvariableop_resource:@@L
>convolutional_block_3_conv2d_3_biasadd_readvariableop_resource:@Q
Cconvolutional_block_3_batch_normalization_3_readvariableop_resource:@S
Econvolutional_block_3_batch_normalization_3_readvariableop_1_resource:@b
Tconvolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource:@d
Vconvolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource:@
identityѕбGconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpбIconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1б6convolutional_block/batch_normalization/ReadVariableOpб8convolutional_block/batch_normalization/ReadVariableOp_1б1convolutional_block/conv2d/BiasAdd/ReadVariableOpб0convolutional_block/conv2d/Conv2D/ReadVariableOpбKconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpбMconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б:convolutional_block_1/batch_normalization_1/ReadVariableOpб<convolutional_block_1/batch_normalization_1/ReadVariableOp_1б5convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOpб4convolutional_block_1/conv2d_1/Conv2D/ReadVariableOpбKconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpбMconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б:convolutional_block_2/batch_normalization_2/ReadVariableOpб<convolutional_block_2/batch_normalization_2/ReadVariableOp_1б5convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOpб4convolutional_block_2/conv2d_2/Conv2D/ReadVariableOpбKconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpбMconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1б:convolutional_block_3/batch_normalization_3/ReadVariableOpб<convolutional_block_3/batch_normalization_3/ReadVariableOp_1б5convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOpб4convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp▓
0convolutional_block/conv2d/Conv2D/ReadVariableOpReadVariableOp9convolutional_block_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0м
!convolutional_block/conv2d/Conv2DConv2Dinputs8convolutional_block/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ *
paddingVALID*
strides
е
1convolutional_block/conv2d/BiasAdd/ReadVariableOpReadVariableOp:convolutional_block_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0л
"convolutional_block/conv2d/BiasAddBiasAdd*convolutional_block/conv2d/Conv2D:output:09convolutional_block/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ ╬
)convolutional_block/max_pooling2d/MaxPoolMaxPool+convolutional_block/conv2d/BiasAdd:output:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
▓
6convolutional_block/batch_normalization/ReadVariableOpReadVariableOp?convolutional_block_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0Х
8convolutional_block/batch_normalization/ReadVariableOp_1ReadVariableOpAconvolutional_block_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0н
Gconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpPconvolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0п
Iconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpRconvolutional_block_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0ф
8convolutional_block/batch_normalization/FusedBatchNormV3FusedBatchNormV32convolutional_block/max_pooling2d/MaxPool:output:0>convolutional_block/batch_normalization/ReadVariableOp:value:0@convolutional_block/batch_normalization/ReadVariableOp_1:value:0Oconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Qconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( ў
convolutional_block/ReluRelu<convolutional_block/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          ║
4convolutional_block_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp=convolutional_block_1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0Э
%convolutional_block_1/conv2d_1/Conv2DConv2D&convolutional_block/Relu:activations:0<convolutional_block_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} *
paddingVALID*
strides
░
5convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp>convolutional_block_1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0┌
&convolutional_block_1/conv2d_1/BiasAddBiasAdd.convolutional_block_1/conv2d_1/Conv2D:output:0=convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} о
-convolutional_block_1/max_pooling2d_1/MaxPoolMaxPool/convolutional_block_1/conv2d_1/BiasAdd:output:0*/
_output_shapes
:         >> *
ksize
*
paddingVALID*
strides
║
:convolutional_block_1/batch_normalization_1/ReadVariableOpReadVariableOpCconvolutional_block_1_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype0Й
<convolutional_block_1/batch_normalization_1/ReadVariableOp_1ReadVariableOpEconvolutional_block_1_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype0▄
Kconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpTconvolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0Я
Mconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVconvolutional_block_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0┬
<convolutional_block_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV36convolutional_block_1/max_pooling2d_1/MaxPool:output:0Bconvolutional_block_1/batch_normalization_1/ReadVariableOp:value:0Dconvolutional_block_1/batch_normalization_1/ReadVariableOp_1:value:0Sconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Uconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         >> : : : : :*
epsilon%oЃ:*
is_training( ъ
convolutional_block_1/ReluRelu@convolutional_block_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         >> ║
4convolutional_block_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp=convolutional_block_2_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0Щ
%convolutional_block_2/conv2d_2/Conv2DConv2D(convolutional_block_1/Relu:activations:0<convolutional_block_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@*
paddingVALID*
strides
░
5convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp>convolutional_block_2_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┌
&convolutional_block_2/conv2d_2/BiasAddBiasAdd.convolutional_block_2/conv2d_2/Conv2D:output:0=convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@о
-convolutional_block_2/max_pooling2d_2/MaxPoolMaxPool/convolutional_block_2/conv2d_2/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
║
:convolutional_block_2/batch_normalization_2/ReadVariableOpReadVariableOpCconvolutional_block_2_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0Й
<convolutional_block_2/batch_normalization_2/ReadVariableOp_1ReadVariableOpEconvolutional_block_2_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0▄
Kconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpTconvolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Я
Mconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVconvolutional_block_2_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0┬
<convolutional_block_2/batch_normalization_2/FusedBatchNormV3FusedBatchNormV36convolutional_block_2/max_pooling2d_2/MaxPool:output:0Bconvolutional_block_2/batch_normalization_2/ReadVariableOp:value:0Dconvolutional_block_2/batch_normalization_2/ReadVariableOp_1:value:0Sconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Uconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( ъ
convolutional_block_2/ReluRelu@convolutional_block_2/batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @║
4convolutional_block_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp=convolutional_block_3_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Щ
%convolutional_block_3/conv2d_3/Conv2DConv2D(convolutional_block_2/Relu:activations:0<convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
░
5convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp>convolutional_block_3_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0┌
&convolutional_block_3/conv2d_3/BiasAddBiasAdd.convolutional_block_3/conv2d_3/Conv2D:output:0=convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @о
-convolutional_block_3/max_pooling2d_3/MaxPoolMaxPool/convolutional_block_3/conv2d_3/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
║
:convolutional_block_3/batch_normalization_3/ReadVariableOpReadVariableOpCconvolutional_block_3_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype0Й
<convolutional_block_3/batch_normalization_3/ReadVariableOp_1ReadVariableOpEconvolutional_block_3_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype0▄
Kconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpTconvolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Я
Mconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpVconvolutional_block_3_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0┬
<convolutional_block_3/batch_normalization_3/FusedBatchNormV3FusedBatchNormV36convolutional_block_3/max_pooling2d_3/MaxPool:output:0Bconvolutional_block_3/batch_normalization_3/ReadVariableOp:value:0Dconvolutional_block_3/batch_normalization_3/ReadVariableOp_1:value:0Sconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Uconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
is_training( ъ
convolutional_block_3/ReluRelu@convolutional_block_3/batch_normalization_3/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @
IdentityIdentity(convolutional_block_3/Relu:activations:0^NoOp*
T0*/
_output_shapes
:         @м
NoOpNoOpH^convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpJ^convolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_17^convolutional_block/batch_normalization/ReadVariableOp9^convolutional_block/batch_normalization/ReadVariableOp_12^convolutional_block/conv2d/BiasAdd/ReadVariableOp1^convolutional_block/conv2d/Conv2D/ReadVariableOpL^convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpN^convolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1;^convolutional_block_1/batch_normalization_1/ReadVariableOp=^convolutional_block_1/batch_normalization_1/ReadVariableOp_16^convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp5^convolutional_block_1/conv2d_1/Conv2D/ReadVariableOpL^convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpN^convolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1;^convolutional_block_2/batch_normalization_2/ReadVariableOp=^convolutional_block_2/batch_normalization_2/ReadVariableOp_16^convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp5^convolutional_block_2/conv2d_2/Conv2D/ReadVariableOpL^convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpN^convolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1;^convolutional_block_3/batch_normalization_3/ReadVariableOp=^convolutional_block_3/batch_normalization_3/ReadVariableOp_16^convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp5^convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : 2њ
Gconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOpGconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp2ќ
Iconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Iconvolutional_block/batch_normalization/FusedBatchNormV3/ReadVariableOp_12p
6convolutional_block/batch_normalization/ReadVariableOp6convolutional_block/batch_normalization/ReadVariableOp2t
8convolutional_block/batch_normalization/ReadVariableOp_18convolutional_block/batch_normalization/ReadVariableOp_12f
1convolutional_block/conv2d/BiasAdd/ReadVariableOp1convolutional_block/conv2d/BiasAdd/ReadVariableOp2d
0convolutional_block/conv2d/Conv2D/ReadVariableOp0convolutional_block/conv2d/Conv2D/ReadVariableOp2џ
Kconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpKconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2ъ
Mconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Mconvolutional_block_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12x
:convolutional_block_1/batch_normalization_1/ReadVariableOp:convolutional_block_1/batch_normalization_1/ReadVariableOp2|
<convolutional_block_1/batch_normalization_1/ReadVariableOp_1<convolutional_block_1/batch_normalization_1/ReadVariableOp_12n
5convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp5convolutional_block_1/conv2d_1/BiasAdd/ReadVariableOp2l
4convolutional_block_1/conv2d_1/Conv2D/ReadVariableOp4convolutional_block_1/conv2d_1/Conv2D/ReadVariableOp2џ
Kconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOpKconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2ъ
Mconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Mconvolutional_block_2/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12x
:convolutional_block_2/batch_normalization_2/ReadVariableOp:convolutional_block_2/batch_normalization_2/ReadVariableOp2|
<convolutional_block_2/batch_normalization_2/ReadVariableOp_1<convolutional_block_2/batch_normalization_2/ReadVariableOp_12n
5convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp5convolutional_block_2/conv2d_2/BiasAdd/ReadVariableOp2l
4convolutional_block_2/conv2d_2/Conv2D/ReadVariableOp4convolutional_block_2/conv2d_2/Conv2D/ReadVariableOp2џ
Kconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOpKconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2ъ
Mconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Mconvolutional_block_3/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12x
:convolutional_block_3/batch_normalization_3/ReadVariableOp:convolutional_block_3/batch_normalization_3/ReadVariableOp2|
<convolutional_block_3/batch_normalization_3/ReadVariableOp_1<convolutional_block_3/batch_normalization_3/ReadVariableOp_12n
5convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp5convolutional_block_3/conv2d_3/BiasAdd/ReadVariableOp2l
4convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp4convolutional_block_3/conv2d_3/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
З
╩
G__inference_sequential_1_layer_call_and_return_conditional_losses_66218
dense_input
dense_66206:	@ђ
dense_66208:	ђ 
dense_1_66212:	ђ
dense_1_66214:
identityѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdropout/StatefulPartitionedCallЖ
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_66206dense_66208*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_66049ж
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_66121ј
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_66212dense_1_66214*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_66073w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ф
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:T P
'
_output_shapes
:         @
%
_user_specified_namedense_input
Ў
Ш
*__inference_sequential_layer_call_fn_67238

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@
identityѕбStatefulPartitionedCallѕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_65802w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Џ&
┘
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_67676

inputsA
'conv2d_1_conv2d_readvariableop_resource:  6
(conv2d_1_biasadd_readvariableop_resource: ;
-batch_normalization_1_readvariableop_resource: =
/batch_normalization_1_readvariableop_1_resource: L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: 
identityѕб$batch_normalization_1/AssignNewValueб&batch_normalization_1/AssignNewValue_1б5batch_normalization_1/FusedBatchNormV3/ReadVariableOpб7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_1/ReadVariableOpб&batch_normalization_1/ReadVariableOp_1бconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpј
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0г
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} *
paddingVALID*
strides
ё
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ў
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} ф
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/BiasAdd:output:0*/
_output_shapes
:         >> *
ksize
*
paddingVALID*
strides
ј
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype0њ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╠
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_1/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         >> : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<ъ
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(е
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(r
ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         >> i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         >> Џ
NoOpNoOp%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':          : : : : : : 2L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
Б	
ј
5__inference_convolutional_block_1_layer_call_fn_67624

inputs!
unknown:  
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identityѕбStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >> *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_65219w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         >> `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':          : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
┼%
р

E__inference_sequential_layer_call_and_return_conditional_losses_66018
convolutional_block_input3
convolutional_block_65965: '
convolutional_block_65967: '
convolutional_block_65969: '
convolutional_block_65971: '
convolutional_block_65973: '
convolutional_block_65975: 5
convolutional_block_1_65978:  )
convolutional_block_1_65980: )
convolutional_block_1_65982: )
convolutional_block_1_65984: )
convolutional_block_1_65986: )
convolutional_block_1_65988: 5
convolutional_block_2_65991: @)
convolutional_block_2_65993:@)
convolutional_block_2_65995:@)
convolutional_block_2_65997:@)
convolutional_block_2_65999:@)
convolutional_block_2_66001:@5
convolutional_block_3_66004:@@)
convolutional_block_3_66006:@)
convolutional_block_3_66008:@)
convolutional_block_3_66010:@)
convolutional_block_3_66012:@)
convolutional_block_3_66014:@
identityѕб+convolutional_block/StatefulPartitionedCallб-convolutional_block_1/StatefulPartitionedCallб-convolutional_block_2/StatefulPartitionedCallб-convolutional_block_3/StatefulPartitionedCallЕ
+convolutional_block/StatefulPartitionedCallStatefulPartitionedCallconvolutional_block_inputconvolutional_block_65965convolutional_block_65967convolutional_block_65969convolutional_block_65971convolutional_block_65973convolutional_block_65975*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_convolutional_block_layer_call_and_return_conditional_losses_64989н
-convolutional_block_1/StatefulPartitionedCallStatefulPartitionedCall4convolutional_block/StatefulPartitionedCall:output:0convolutional_block_1_65978convolutional_block_1_65980convolutional_block_1_65982convolutional_block_1_65984convolutional_block_1_65986convolutional_block_1_65988*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >> *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_65219о
-convolutional_block_2/StatefulPartitionedCallStatefulPartitionedCall6convolutional_block_1/StatefulPartitionedCall:output:0convolutional_block_2_65991convolutional_block_2_65993convolutional_block_2_65995convolutional_block_2_65997convolutional_block_2_65999convolutional_block_2_66001*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_65392о
-convolutional_block_3/StatefulPartitionedCallStatefulPartitionedCall6convolutional_block_2/StatefulPartitionedCall:output:0convolutional_block_3_66004convolutional_block_3_66006convolutional_block_3_66008convolutional_block_3_66010convolutional_block_3_66012convolutional_block_3_66014*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_65565Ї
IdentityIdentity6convolutional_block_3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @ё
NoOpNoOp,^convolutional_block/StatefulPartitionedCall.^convolutional_block_1/StatefulPartitionedCall.^convolutional_block_2/StatefulPartitionedCall.^convolutional_block_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+convolutional_block/StatefulPartitionedCall+convolutional_block/StatefulPartitionedCall2^
-convolutional_block_1/StatefulPartitionedCall-convolutional_block_1/StatefulPartitionedCall2^
-convolutional_block_2/StatefulPartitionedCall-convolutional_block_2/StatefulPartitionedCall2^
-convolutional_block_3/StatefulPartitionedCall-convolutional_block_3/StatefulPartitionedCall:l h
1
_output_shapes
:         ђђ
3
_user_specified_nameconvolutional_block_input
Ё
┐
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68279

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
╩
§
N__inference_convolutional_model_layer_call_and_return_conditional_losses_66716
input_1*
sequential_66656: 
sequential_66658: 
sequential_66660: 
sequential_66662: 
sequential_66664: 
sequential_66666: *
sequential_66668:  
sequential_66670: 
sequential_66672: 
sequential_66674: 
sequential_66676: 
sequential_66678: *
sequential_66680: @
sequential_66682:@
sequential_66684:@
sequential_66686:@
sequential_66688:@
sequential_66690:@*
sequential_66692:@@
sequential_66694:@
sequential_66696:@
sequential_66698:@
sequential_66700:@
sequential_66702:@%
sequential_1_66706:	@ђ!
sequential_1_66708:	ђ%
sequential_1_66710:	ђ 
sequential_1_66712:
identityѕб"sequential/StatefulPartitionedCallб$sequential_1/StatefulPartitionedCall▒
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_66656sequential_66658sequential_66660sequential_66662sequential_66664sequential_66666sequential_66668sequential_66670sequential_66672sequential_66674sequential_66676sequential_66678sequential_66680sequential_66682sequential_66684sequential_66686sequential_66688sequential_66690sequential_66692sequential_66694sequential_66696sequential_66698sequential_66700sequential_66702*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_65802 
(global_average_pooling2d/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_66028О
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0sequential_1_66706sequential_1_66708sequential_1_66710sequential_1_66712*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_66164|
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         њ
NoOpNoOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1
юб
¤+
__inference__traced_save_68533
file_prefix@
<savev2_convolutional_block_conv2d_kernel_read_readvariableop>
:savev2_convolutional_block_conv2d_bias_read_readvariableopL
Hsavev2_convolutional_block_batch_normalization_gamma_read_readvariableopK
Gsavev2_convolutional_block_batch_normalization_beta_read_readvariableopR
Nsavev2_convolutional_block_batch_normalization_moving_mean_read_readvariableopV
Rsavev2_convolutional_block_batch_normalization_moving_variance_read_readvariableopD
@savev2_convolutional_block_1_conv2d_1_kernel_read_readvariableopB
>savev2_convolutional_block_1_conv2d_1_bias_read_readvariableopP
Lsavev2_convolutional_block_1_batch_normalization_1_gamma_read_readvariableopO
Ksavev2_convolutional_block_1_batch_normalization_1_beta_read_readvariableopV
Rsavev2_convolutional_block_1_batch_normalization_1_moving_mean_read_readvariableopZ
Vsavev2_convolutional_block_1_batch_normalization_1_moving_variance_read_readvariableopD
@savev2_convolutional_block_2_conv2d_2_kernel_read_readvariableopB
>savev2_convolutional_block_2_conv2d_2_bias_read_readvariableopP
Lsavev2_convolutional_block_2_batch_normalization_2_gamma_read_readvariableopO
Ksavev2_convolutional_block_2_batch_normalization_2_beta_read_readvariableopV
Rsavev2_convolutional_block_2_batch_normalization_2_moving_mean_read_readvariableopZ
Vsavev2_convolutional_block_2_batch_normalization_2_moving_variance_read_readvariableopD
@savev2_convolutional_block_3_conv2d_3_kernel_read_readvariableopB
>savev2_convolutional_block_3_conv2d_3_bias_read_readvariableopP
Lsavev2_convolutional_block_3_batch_normalization_3_gamma_read_readvariableopO
Ksavev2_convolutional_block_3_batch_normalization_3_beta_read_readvariableopV
Rsavev2_convolutional_block_3_batch_normalization_3_moving_mean_read_readvariableopZ
Vsavev2_convolutional_block_3_batch_normalization_3_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopG
Csavev2_adam_convolutional_block_conv2d_kernel_m_read_readvariableopE
Asavev2_adam_convolutional_block_conv2d_bias_m_read_readvariableopS
Osavev2_adam_convolutional_block_batch_normalization_gamma_m_read_readvariableopR
Nsavev2_adam_convolutional_block_batch_normalization_beta_m_read_readvariableopK
Gsavev2_adam_convolutional_block_1_conv2d_1_kernel_m_read_readvariableopI
Esavev2_adam_convolutional_block_1_conv2d_1_bias_m_read_readvariableopW
Ssavev2_adam_convolutional_block_1_batch_normalization_1_gamma_m_read_readvariableopV
Rsavev2_adam_convolutional_block_1_batch_normalization_1_beta_m_read_readvariableopK
Gsavev2_adam_convolutional_block_2_conv2d_2_kernel_m_read_readvariableopI
Esavev2_adam_convolutional_block_2_conv2d_2_bias_m_read_readvariableopW
Ssavev2_adam_convolutional_block_2_batch_normalization_2_gamma_m_read_readvariableopV
Rsavev2_adam_convolutional_block_2_batch_normalization_2_beta_m_read_readvariableopK
Gsavev2_adam_convolutional_block_3_conv2d_3_kernel_m_read_readvariableopI
Esavev2_adam_convolutional_block_3_conv2d_3_bias_m_read_readvariableopW
Ssavev2_adam_convolutional_block_3_batch_normalization_3_gamma_m_read_readvariableopV
Rsavev2_adam_convolutional_block_3_batch_normalization_3_beta_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableopG
Csavev2_adam_convolutional_block_conv2d_kernel_v_read_readvariableopE
Asavev2_adam_convolutional_block_conv2d_bias_v_read_readvariableopS
Osavev2_adam_convolutional_block_batch_normalization_gamma_v_read_readvariableopR
Nsavev2_adam_convolutional_block_batch_normalization_beta_v_read_readvariableopK
Gsavev2_adam_convolutional_block_1_conv2d_1_kernel_v_read_readvariableopI
Esavev2_adam_convolutional_block_1_conv2d_1_bias_v_read_readvariableopW
Ssavev2_adam_convolutional_block_1_batch_normalization_1_gamma_v_read_readvariableopV
Rsavev2_adam_convolutional_block_1_batch_normalization_1_beta_v_read_readvariableopK
Gsavev2_adam_convolutional_block_2_conv2d_2_kernel_v_read_readvariableopI
Esavev2_adam_convolutional_block_2_conv2d_2_bias_v_read_readvariableopW
Ssavev2_adam_convolutional_block_2_batch_normalization_2_gamma_v_read_readvariableopV
Rsavev2_adam_convolutional_block_2_batch_normalization_2_beta_v_read_readvariableopK
Gsavev2_adam_convolutional_block_3_conv2d_3_kernel_v_read_readvariableopI
Esavev2_adam_convolutional_block_3_conv2d_3_bias_v_read_readvariableopW
Ssavev2_adam_convolutional_block_3_batch_normalization_3_gamma_v_read_readvariableopV
Rsavev2_adam_convolutional_block_3_batch_normalization_3_beta_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partЂ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ы"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*џ"
valueљ"BЇ"NB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/26/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/27/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHї
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:N*
dtype0*▒
valueДBцNB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Г*
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0<savev2_convolutional_block_conv2d_kernel_read_readvariableop:savev2_convolutional_block_conv2d_bias_read_readvariableopHsavev2_convolutional_block_batch_normalization_gamma_read_readvariableopGsavev2_convolutional_block_batch_normalization_beta_read_readvariableopNsavev2_convolutional_block_batch_normalization_moving_mean_read_readvariableopRsavev2_convolutional_block_batch_normalization_moving_variance_read_readvariableop@savev2_convolutional_block_1_conv2d_1_kernel_read_readvariableop>savev2_convolutional_block_1_conv2d_1_bias_read_readvariableopLsavev2_convolutional_block_1_batch_normalization_1_gamma_read_readvariableopKsavev2_convolutional_block_1_batch_normalization_1_beta_read_readvariableopRsavev2_convolutional_block_1_batch_normalization_1_moving_mean_read_readvariableopVsavev2_convolutional_block_1_batch_normalization_1_moving_variance_read_readvariableop@savev2_convolutional_block_2_conv2d_2_kernel_read_readvariableop>savev2_convolutional_block_2_conv2d_2_bias_read_readvariableopLsavev2_convolutional_block_2_batch_normalization_2_gamma_read_readvariableopKsavev2_convolutional_block_2_batch_normalization_2_beta_read_readvariableopRsavev2_convolutional_block_2_batch_normalization_2_moving_mean_read_readvariableopVsavev2_convolutional_block_2_batch_normalization_2_moving_variance_read_readvariableop@savev2_convolutional_block_3_conv2d_3_kernel_read_readvariableop>savev2_convolutional_block_3_conv2d_3_bias_read_readvariableopLsavev2_convolutional_block_3_batch_normalization_3_gamma_read_readvariableopKsavev2_convolutional_block_3_batch_normalization_3_beta_read_readvariableopRsavev2_convolutional_block_3_batch_normalization_3_moving_mean_read_readvariableopVsavev2_convolutional_block_3_batch_normalization_3_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopCsavev2_adam_convolutional_block_conv2d_kernel_m_read_readvariableopAsavev2_adam_convolutional_block_conv2d_bias_m_read_readvariableopOsavev2_adam_convolutional_block_batch_normalization_gamma_m_read_readvariableopNsavev2_adam_convolutional_block_batch_normalization_beta_m_read_readvariableopGsavev2_adam_convolutional_block_1_conv2d_1_kernel_m_read_readvariableopEsavev2_adam_convolutional_block_1_conv2d_1_bias_m_read_readvariableopSsavev2_adam_convolutional_block_1_batch_normalization_1_gamma_m_read_readvariableopRsavev2_adam_convolutional_block_1_batch_normalization_1_beta_m_read_readvariableopGsavev2_adam_convolutional_block_2_conv2d_2_kernel_m_read_readvariableopEsavev2_adam_convolutional_block_2_conv2d_2_bias_m_read_readvariableopSsavev2_adam_convolutional_block_2_batch_normalization_2_gamma_m_read_readvariableopRsavev2_adam_convolutional_block_2_batch_normalization_2_beta_m_read_readvariableopGsavev2_adam_convolutional_block_3_conv2d_3_kernel_m_read_readvariableopEsavev2_adam_convolutional_block_3_conv2d_3_bias_m_read_readvariableopSsavev2_adam_convolutional_block_3_batch_normalization_3_gamma_m_read_readvariableopRsavev2_adam_convolutional_block_3_batch_normalization_3_beta_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableopCsavev2_adam_convolutional_block_conv2d_kernel_v_read_readvariableopAsavev2_adam_convolutional_block_conv2d_bias_v_read_readvariableopOsavev2_adam_convolutional_block_batch_normalization_gamma_v_read_readvariableopNsavev2_adam_convolutional_block_batch_normalization_beta_v_read_readvariableopGsavev2_adam_convolutional_block_1_conv2d_1_kernel_v_read_readvariableopEsavev2_adam_convolutional_block_1_conv2d_1_bias_v_read_readvariableopSsavev2_adam_convolutional_block_1_batch_normalization_1_gamma_v_read_readvariableopRsavev2_adam_convolutional_block_1_batch_normalization_1_beta_v_read_readvariableopGsavev2_adam_convolutional_block_2_conv2d_2_kernel_v_read_readvariableopEsavev2_adam_convolutional_block_2_conv2d_2_bias_v_read_readvariableopSsavev2_adam_convolutional_block_2_batch_normalization_2_gamma_v_read_readvariableopRsavev2_adam_convolutional_block_2_batch_normalization_2_beta_v_read_readvariableopGsavev2_adam_convolutional_block_3_conv2d_3_kernel_v_read_readvariableopEsavev2_adam_convolutional_block_3_conv2d_3_bias_v_read_readvariableopSsavev2_adam_convolutional_block_3_batch_normalization_3_gamma_v_read_readvariableopRsavev2_adam_convolutional_block_3_batch_normalization_3_beta_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *\
dtypesR
P2N	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*З
_input_shapesР
▀: : : : : : : :  : : : : : : @:@:@:@:@:@:@@:@:@:@:@:@:	@ђ:ђ:	ђ:: : : : : : : : : : : : : :  : : : : @:@:@:@:@@:@:@:@:	@ђ:ђ:	ђ:: : : : :  : : : : @:@:@:@:@@:@:@:@:	@ђ:ђ:	ђ:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:%!

_output_shapes
:	@ђ:!

_output_shapes	
:ђ:%!

_output_shapes
:	ђ: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :,&(
&
_output_shapes
: : '

_output_shapes
: : (

_output_shapes
: : )

_output_shapes
: :,*(
&
_output_shapes
:  : +

_output_shapes
: : ,

_output_shapes
: : -

_output_shapes
: :,.(
&
_output_shapes
: @: /

_output_shapes
:@: 0

_output_shapes
:@: 1

_output_shapes
:@:,2(
&
_output_shapes
:@@: 3

_output_shapes
:@: 4

_output_shapes
:@: 5

_output_shapes
:@:%6!

_output_shapes
:	@ђ:!7

_output_shapes	
:ђ:%8!

_output_shapes
:	ђ: 9

_output_shapes
::,:(
&
_output_shapes
: : ;

_output_shapes
: : <

_output_shapes
: : =

_output_shapes
: :,>(
&
_output_shapes
:  : ?

_output_shapes
: : @

_output_shapes
: : A

_output_shapes
: :,B(
&
_output_shapes
: @: C

_output_shapes
:@: D

_output_shapes
:@: E

_output_shapes
:@:,F(
&
_output_shapes
:@@: G

_output_shapes
:@: H

_output_shapes
:@: I

_output_shapes
:@:%J!

_output_shapes
:	@ђ:!K

_output_shapes	
:ђ:%L!

_output_shapes
:	ђ: M

_output_shapes
::N

_output_shapes
: 
Д

Ч
C__inference_conv2d_1_layer_call_and_return_conditional_losses_65154

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         }} w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:          : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
▒
О
,__inference_sequential_1_layer_call_fn_66188
dense_input
unknown:	@ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:
identityѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_66164o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:         @
%
_user_specified_namedense_input
т
┼
G__inference_sequential_1_layer_call_and_return_conditional_losses_66164

inputs
dense_66152:	@ђ
dense_66154:	ђ 
dense_1_66158:	ђ
dense_1_66160:
identityѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallбdropout/StatefulPartitionedCallт
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_66152dense_66154*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_66049ж
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_66121ј
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_1_66158dense_1_66160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_66073w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ф
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╔
Ў
N__inference_batch_normalization_layer_call_and_return_conditional_losses_64848

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
╬
Ы
3__inference_convolutional_model_layer_call_fn_66907

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	@ђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:
identityѕбStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_convolutional_model_layer_call_and_return_conditional_losses_66470o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
┘
`
B__inference_dropout_layer_call_and_return_conditional_losses_66060

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         ђ\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ђ"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
љ	
л
5__inference_batch_normalization_2_layer_call_fn_68152

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallџ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_65299Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ђ
Ѕ
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_67650

inputsA
'conv2d_1_conv2d_readvariableop_resource:  6
(conv2d_1_biasadd_readvariableop_resource: ;
-batch_normalization_1_readvariableop_resource: =
/batch_normalization_1_readvariableop_1_resource: L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource: N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource: 
identityѕб5batch_normalization_1/FusedBatchNormV3/ReadVariableOpб7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_1/ReadVariableOpб&batch_normalization_1/ReadVariableOp_1бconv2d_1/BiasAdd/ReadVariableOpбconv2d_1/Conv2D/ReadVariableOpј
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype0г
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} *
paddingVALID*
strides
ё
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ў
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         }} ф
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/BiasAdd:output:0*/
_output_shapes
:         >> *
ksize
*
paddingVALID*
strides
ј
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype0њ
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype0░
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0┤
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0Й
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_1/MaxPool:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         >> : : : : :*
epsilon%oЃ:*
is_training( r
ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         >> i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         >> ╦
NoOpNoOp6^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1 ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':          : : : : : : 2n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
»

Щ
A__inference_conv2d_layer_call_and_return_conditional_losses_67934

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ю
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ *
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ i
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:         ■■ w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:         ђђ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Ѓ
й
N__inference_batch_normalization_layer_call_and_return_conditional_losses_68006

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ѓ
й
N__inference_batch_normalization_layer_call_and_return_conditional_losses_64879

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
ј
ц
N__inference_convolutional_block_layer_call_and_return_conditional_losses_64925

inputs&
conv2d_64908: 
conv2d_64910: '
batch_normalization_64914: '
batch_normalization_64916: '
batch_normalization_64918: '
batch_normalization_64920: 
identityѕб+batch_normalization/StatefulPartitionedCallбconv2d/StatefulPartitionedCallЫ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_64908conv2d_64910*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ■■ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_64907ь
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_64823■
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_64914batch_normalization_64916batch_normalization_64918batch_normalization_64920*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_64848|
ReluRelu4batch_normalization/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          Ћ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
К
и
N__inference_convolutional_block_layer_call_and_return_conditional_losses_65041
convolutional_block_input&
conv2d_65024: 
conv2d_65026: '
batch_normalization_65030: '
batch_normalization_65032: '
batch_normalization_65034: '
batch_normalization_65036: 
identityѕб+batch_normalization/StatefulPartitionedCallбconv2d/StatefulPartitionedCallЁ
conv2d/StatefulPartitionedCallStatefulPartitionedCallconvolutional_block_inputconv2d_65024conv2d_65026*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:         ■■ *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_64907ь
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *Q
fLRJ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_64823■
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0batch_normalization_65030batch_normalization_65032batch_normalization_65034batch_normalization_65036*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_64848|
ReluRelu4batch_normalization/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          Ћ
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall:l h
1
_output_shapes
:         ђђ
3
_user_specified_nameconvolutional_block_input
┘
з
3__inference_convolutional_model_layer_call_fn_66344
input_1!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: #
	unknown_5:  
	unknown_6: 
	unknown_7: 
	unknown_8: 
	unknown_9: 

unknown_10: $

unknown_11: @

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@$

unknown_17:@@

unknown_18:@

unknown_19:@

unknown_20:@

unknown_21:@

unknown_22:@

unknown_23:	@ђ

unknown_24:	ђ

unknown_25:	ђ

unknown_26:
identityѕбStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *>
_read_only_resource_inputs 
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_convolutional_model_layer_call_and_return_conditional_losses_66285o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1
┬
Х
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_65565

inputs(
conv2d_3_65548:@@
conv2d_3_65550:@)
batch_normalization_3_65554:@)
batch_normalization_3_65556:@)
batch_normalization_3_65558:@)
batch_normalization_3_65560:@
identityѕб-batch_normalization_3/StatefulPartitionedCallб conv2d_3/StatefulPartitionedCallЭ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_65548conv2d_3_65550*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_65500з
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_65416і
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_3/PartitionedCall:output:0batch_normalization_3_65554batch_normalization_3_65556batch_normalization_3_65558batch_normalization_3_65560*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_65472~
ReluRelu6batch_normalization_3/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @Ў
NoOpNoOp.^batch_normalization_3/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @: : : : : : 2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
╦
Џ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68079

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                            : : : : :*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                            ░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
═%
р

E__inference_sequential_layer_call_and_return_conditional_losses_65962
convolutional_block_input3
convolutional_block_65909: '
convolutional_block_65911: '
convolutional_block_65913: '
convolutional_block_65915: '
convolutional_block_65917: '
convolutional_block_65919: 5
convolutional_block_1_65922:  )
convolutional_block_1_65924: )
convolutional_block_1_65926: )
convolutional_block_1_65928: )
convolutional_block_1_65930: )
convolutional_block_1_65932: 5
convolutional_block_2_65935: @)
convolutional_block_2_65937:@)
convolutional_block_2_65939:@)
convolutional_block_2_65941:@)
convolutional_block_2_65943:@)
convolutional_block_2_65945:@5
convolutional_block_3_65948:@@)
convolutional_block_3_65950:@)
convolutional_block_3_65952:@)
convolutional_block_3_65954:@)
convolutional_block_3_65956:@)
convolutional_block_3_65958:@
identityѕб+convolutional_block/StatefulPartitionedCallб-convolutional_block_1/StatefulPartitionedCallб-convolutional_block_2/StatefulPartitionedCallб-convolutional_block_3/StatefulPartitionedCallФ
+convolutional_block/StatefulPartitionedCallStatefulPartitionedCallconvolutional_block_inputconvolutional_block_65909convolutional_block_65911convolutional_block_65913convolutional_block_65915convolutional_block_65917convolutional_block_65919*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_convolutional_block_layer_call_and_return_conditional_losses_64925о
-convolutional_block_1/StatefulPartitionedCallStatefulPartitionedCall4convolutional_block/StatefulPartitionedCall:output:0convolutional_block_1_65922convolutional_block_1_65924convolutional_block_1_65926convolutional_block_1_65928convolutional_block_1_65930convolutional_block_1_65932*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >> *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_65172п
-convolutional_block_2/StatefulPartitionedCallStatefulPartitionedCall6convolutional_block_1/StatefulPartitionedCall:output:0convolutional_block_2_65935convolutional_block_2_65937convolutional_block_2_65939convolutional_block_2_65941convolutional_block_2_65943convolutional_block_2_65945*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_65345п
-convolutional_block_3/StatefulPartitionedCallStatefulPartitionedCall6convolutional_block_2/StatefulPartitionedCall:output:0convolutional_block_3_65948convolutional_block_3_65950convolutional_block_3_65952convolutional_block_3_65954convolutional_block_3_65956convolutional_block_3_65958*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_65518Ї
IdentityIdentity6convolutional_block_3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @ё
NoOpNoOp,^convolutional_block/StatefulPartitionedCall.^convolutional_block_1/StatefulPartitionedCall.^convolutional_block_2/StatefulPartitionedCall.^convolutional_block_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+convolutional_block/StatefulPartitionedCall+convolutional_block/StatefulPartitionedCall2^
-convolutional_block_1/StatefulPartitionedCall-convolutional_block_1/StatefulPartitionedCall2^
-convolutional_block_2/StatefulPartitionedCall-convolutional_block_2/StatefulPartitionedCall2^
-convolutional_block_3/StatefulPartitionedCall-convolutional_block_3/StatefulPartitionedCall:l h
1
_output_shapes
:         ђђ
3
_user_specified_nameconvolutional_block_input
м
§
N__inference_convolutional_model_layer_call_and_return_conditional_losses_66653
input_1*
sequential_66593: 
sequential_66595: 
sequential_66597: 
sequential_66599: 
sequential_66601: 
sequential_66603: *
sequential_66605:  
sequential_66607: 
sequential_66609: 
sequential_66611: 
sequential_66613: 
sequential_66615: *
sequential_66617: @
sequential_66619:@
sequential_66621:@
sequential_66623:@
sequential_66625:@
sequential_66627:@*
sequential_66629:@@
sequential_66631:@
sequential_66633:@
sequential_66635:@
sequential_66637:@
sequential_66639:@%
sequential_1_66643:	@ђ!
sequential_1_66645:	ђ%
sequential_1_66647:	ђ 
sequential_1_66649:
identityѕб"sequential/StatefulPartitionedCallб$sequential_1/StatefulPartitionedCall╣
"sequential/StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_66593sequential_66595sequential_66597sequential_66599sequential_66601sequential_66603sequential_66605sequential_66607sequential_66609sequential_66611sequential_66613sequential_66615sequential_66617sequential_66619sequential_66621sequential_66623sequential_66625sequential_66627sequential_66629sequential_66631sequential_66633sequential_66635sequential_66637sequential_66639*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_65640 
(global_average_pooling2d/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_66028О
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0sequential_1_66643sequential_1_66645sequential_1_66647sequential_1_66649*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_66080|
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         њ
NoOpNoOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:Z V
1
_output_shapes
:         ђђ
!
_user_specified_name	input_1
ћ%
╬

E__inference_sequential_layer_call_and_return_conditional_losses_65640

inputs3
convolutional_block_65587: '
convolutional_block_65589: '
convolutional_block_65591: '
convolutional_block_65593: '
convolutional_block_65595: '
convolutional_block_65597: 5
convolutional_block_1_65600:  )
convolutional_block_1_65602: )
convolutional_block_1_65604: )
convolutional_block_1_65606: )
convolutional_block_1_65608: )
convolutional_block_1_65610: 5
convolutional_block_2_65613: @)
convolutional_block_2_65615:@)
convolutional_block_2_65617:@)
convolutional_block_2_65619:@)
convolutional_block_2_65621:@)
convolutional_block_2_65623:@5
convolutional_block_3_65626:@@)
convolutional_block_3_65628:@)
convolutional_block_3_65630:@)
convolutional_block_3_65632:@)
convolutional_block_3_65634:@)
convolutional_block_3_65636:@
identityѕб+convolutional_block/StatefulPartitionedCallб-convolutional_block_1/StatefulPartitionedCallб-convolutional_block_2/StatefulPartitionedCallб-convolutional_block_3/StatefulPartitionedCallў
+convolutional_block/StatefulPartitionedCallStatefulPartitionedCallinputsconvolutional_block_65587convolutional_block_65589convolutional_block_65591convolutional_block_65593convolutional_block_65595convolutional_block_65597*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_convolutional_block_layer_call_and_return_conditional_losses_64925о
-convolutional_block_1/StatefulPartitionedCallStatefulPartitionedCall4convolutional_block/StatefulPartitionedCall:output:0convolutional_block_1_65600convolutional_block_1_65602convolutional_block_1_65604convolutional_block_1_65606convolutional_block_1_65608convolutional_block_1_65610*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >> *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_65172п
-convolutional_block_2/StatefulPartitionedCallStatefulPartitionedCall6convolutional_block_1/StatefulPartitionedCall:output:0convolutional_block_2_65613convolutional_block_2_65615convolutional_block_2_65617convolutional_block_2_65619convolutional_block_2_65621convolutional_block_2_65623*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_65345п
-convolutional_block_3/StatefulPartitionedCallStatefulPartitionedCall6convolutional_block_2/StatefulPartitionedCall:output:0convolutional_block_3_65626convolutional_block_3_65628convolutional_block_3_65630convolutional_block_3_65632convolutional_block_3_65634convolutional_block_3_65636*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_65518Ї
IdentityIdentity6convolutional_block_3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @ё
NoOpNoOp,^convolutional_block/StatefulPartitionedCall.^convolutional_block_1/StatefulPartitionedCall.^convolutional_block_2/StatefulPartitionedCall.^convolutional_block_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*`
_input_shapesO
M:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : 2Z
+convolutional_block/StatefulPartitionedCall+convolutional_block/StatefulPartitionedCall2^
-convolutional_block_1/StatefulPartitionedCall-convolutional_block_1/StatefulPartitionedCall2^
-convolutional_block_2/StatefulPartitionedCall-convolutional_block_2/StatefulPartitionedCall2^
-convolutional_block_3/StatefulPartitionedCall-convolutional_block_3/StatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
њ
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_65070

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
Ц	
ј
5__inference_convolutional_block_3_layer_call_fn_67779

inputs!
unknown:@@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identityѕбStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_65518w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         @: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Б	
ј
5__inference_convolutional_block_2_layer_call_fn_67710

inputs!
unknown: @
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
identityѕбStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_65392w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         >> : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:         >> 
 
_user_specified_nameinputs
К
Ч
N__inference_convolutional_model_layer_call_and_return_conditional_losses_66470

inputs*
sequential_66410: 
sequential_66412: 
sequential_66414: 
sequential_66416: 
sequential_66418: 
sequential_66420: *
sequential_66422:  
sequential_66424: 
sequential_66426: 
sequential_66428: 
sequential_66430: 
sequential_66432: *
sequential_66434: @
sequential_66436:@
sequential_66438:@
sequential_66440:@
sequential_66442:@
sequential_66444:@*
sequential_66446:@@
sequential_66448:@
sequential_66450:@
sequential_66452:@
sequential_66454:@
sequential_66456:@%
sequential_1_66460:	@ђ!
sequential_1_66462:	ђ%
sequential_1_66464:	ђ 
sequential_1_66466:
identityѕб"sequential/StatefulPartitionedCallб$sequential_1/StatefulPartitionedCall░
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_66410sequential_66412sequential_66414sequential_66416sequential_66418sequential_66420sequential_66422sequential_66424sequential_66426sequential_66428sequential_66430sequential_66432sequential_66434sequential_66436sequential_66438sequential_66440sequential_66442sequential_66444sequential_66446sequential_66448sequential_66450sequential_66452sequential_66454sequential_66456*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_65802 
(global_average_pooling2d/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_66028О
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0sequential_1_66460sequential_1_66462sequential_1_66464sequential_1_66466*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_66164|
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         њ
NoOpNoOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Ц	
ї
3__inference_convolutional_block_layer_call_fn_67521

inputs!
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
	unknown_3: 
	unknown_4: 
identityѕбStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:          *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_convolutional_block_layer_call_and_return_conditional_losses_64925w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:          `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
њ
f
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_65243

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╣
K
/__inference_max_pooling2d_2_layer_call_fn_68121

inputs
identity█
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4                                    * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_65243Ѓ
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
╦
Џ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68261

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╚
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @░
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ї	
╬
3__inference_batch_normalization_layer_call_fn_67970

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identityѕбStatefulPartitionedCallў
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                            *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_64879Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                            `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                            : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                            
 
_user_specified_nameinputs
Ы
`
'__inference_dropout_layer_call_fn_67878

inputs
identityѕбStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_66121p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         ђ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
┼
Б
G__inference_sequential_1_layer_call_and_return_conditional_losses_66080

inputs
dense_66050:	@ђ
dense_66052:	ђ 
dense_1_66074:	ђ
dense_1_66076:
identityѕбdense/StatefulPartitionedCallбdense_1/StatefulPartitionedCallт
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_66050dense_66052*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_66049┘
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         ђ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_66060є
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_1_66074dense_1_66076*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_66073w
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ѕ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Ё
┐
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_65472

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identityѕбAssignNewValueбAssignNewValue_1бFusedBatchNormV3/ReadVariableOpб!FusedBatchNormV3/ReadVariableOp_1бReadVariableOpбReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0ё
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0ѕ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0о
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+                           @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<к
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(л
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+                           @н
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
Ќ
№
N__inference_convolutional_block_layer_call_and_return_conditional_losses_67564

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: 9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: 
identityѕб3batch_normalization/FusedBatchNormV3/ReadVariableOpб5batch_normalization/FusedBatchNormV3/ReadVariableOp_1б"batch_normalization/ReadVariableOpб$batch_normalization/ReadVariableOp_1бconv2d/BiasAdd/ReadVariableOpбconv2d/Conv2D/ReadVariableOpі
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0ф
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ *
paddingVALID*
strides
ђ
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0ћ
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:         ■■ д
max_pooling2d/MaxPoolMaxPoolconv2d/BiasAdd:output:0*/
_output_shapes
:          *
ksize
*
paddingVALID*
strides
і
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0ј
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0г
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0░
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0▓
$batch_normalization/FusedBatchNormV3FusedBatchNormV3max_pooling2d/MaxPool:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:          : : : : :*
epsilon%oЃ:*
is_training( p
ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:          i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:          ┐
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         ђђ: : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Д

Ч
C__inference_conv2d_3_layer_call_and_return_conditional_losses_68207

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
Д

Ч
C__inference_conv2d_3_layer_call_and_return_conditional_losses_65500

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбConv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0џ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         @: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:         @
 
_user_specified_nameinputs
њ
f
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_68035

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
┤
o
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_67433

inputs
identityg
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      p
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:                  ^
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
њ
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_65416

inputs
identityб
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4                                    *
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4                                    "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4                                    :r n
J
_output_shapes8
6:4                                    
 
_user_specified_nameinputs
б
м
,__inference_sequential_1_layer_call_fn_67446

inputs
unknown:	@ђ
	unknown_0:	ђ
	unknown_1:	ђ
	unknown_2:
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_66080o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
┬
Х
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_65219

inputs(
conv2d_1_65202:  
conv2d_1_65204: )
batch_normalization_1_65208: )
batch_normalization_1_65210: )
batch_normalization_1_65212: )
batch_normalization_1_65214: 
identityѕб-batch_normalization_1/StatefulPartitionedCallб conv2d_1/StatefulPartitionedCallЭ
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_65202conv2d_1_65204*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         }} *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_65154з
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >> * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_65070і
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0batch_normalization_1_65208batch_normalization_1_65210batch_normalization_1_65212batch_normalization_1_65214*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         >> *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_65126~
ReluRelu6batch_normalization_1/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:         >> i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         >> Ў
NoOpNoOp.^batch_normalization_1/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':          : : : : : : 2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:          
 
_user_specified_nameinputs
њ	
л
5__inference_batch_normalization_2_layer_call_fn_68139

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identityѕбStatefulPartitionedCallю
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+                           @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_65268Ѕ
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+                           @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+                           @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+                           @
 
_user_specified_nameinputs
ё

a
B__inference_dropout_layer_call_and_return_conditional_losses_67895

inputs
identityѕR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n█Х?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ђC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Ў
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ђ*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *џЎЎ>Д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ђp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ђj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ђZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ђ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         ђ:P L
(
_output_shapes
:         ђ
 
_user_specified_nameinputs
¤
Ч
N__inference_convolutional_model_layer_call_and_return_conditional_losses_66285

inputs*
sequential_66225: 
sequential_66227: 
sequential_66229: 
sequential_66231: 
sequential_66233: 
sequential_66235: *
sequential_66237:  
sequential_66239: 
sequential_66241: 
sequential_66243: 
sequential_66245: 
sequential_66247: *
sequential_66249: @
sequential_66251:@
sequential_66253:@
sequential_66255:@
sequential_66257:@
sequential_66259:@*
sequential_66261:@@
sequential_66263:@
sequential_66265:@
sequential_66267:@
sequential_66269:@
sequential_66271:@%
sequential_1_66275:	@ђ!
sequential_1_66277:	ђ%
sequential_1_66279:	ђ 
sequential_1_66281:
identityѕб"sequential/StatefulPartitionedCallб$sequential_1/StatefulPartitionedCallИ
"sequential/StatefulPartitionedCallStatefulPartitionedCallinputssequential_66225sequential_66227sequential_66229sequential_66231sequential_66233sequential_66235sequential_66237sequential_66239sequential_66241sequential_66243sequential_66245sequential_66247sequential_66249sequential_66251sequential_66253sequential_66255sequential_66257sequential_66259sequential_66261sequential_66263sequential_66265sequential_66267sequential_66269sequential_66271*$
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:         @*:
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_65640 
(global_average_pooling2d/PartitionedCallPartitionedCall+sequential/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_66028О
$sequential_1/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0sequential_1_66275sequential_1_66277sequential_1_66279sequential_1_66281*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *P
fKRI
G__inference_sequential_1_layer_call_and_return_conditional_losses_66080|
IdentityIdentity-sequential_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         њ
NoOpNoOp#^sequential/StatefulPartitionedCall%^sequential_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*h
_input_shapesW
U:         ђђ: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall2L
$sequential_1/StatefulPartitionedCall$sequential_1/StatefulPartitionedCall:Y U
1
_output_shapes
:         ђђ
 
_user_specified_nameinputs
Џ&
┘
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_67762

inputsA
'conv2d_2_conv2d_readvariableop_resource: @6
(conv2d_2_biasadd_readvariableop_resource:@;
-batch_normalization_2_readvariableop_resource:@=
/batch_normalization_2_readvariableop_1_resource:@L
>batch_normalization_2_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource:@
identityѕб$batch_normalization_2/AssignNewValueб&batch_normalization_2/AssignNewValue_1б5batch_normalization_2/FusedBatchNormV3/ReadVariableOpб7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1б$batch_normalization_2/ReadVariableOpб&batch_normalization_2/ReadVariableOp_1бconv2d_2/BiasAdd/ReadVariableOpбconv2d_2/Conv2D/ReadVariableOpј
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0г
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@*
paddingVALID*
strides
ё
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ў
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:         <<@ф
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/BiasAdd:output:0*/
_output_shapes
:         @*
ksize
*
paddingVALID*
strides
ј
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype0њ
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype0░
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0┤
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0╠
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3 max_pooling2d_2/MaxPool:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:         @:@:@:@:@:*
epsilon%oЃ:*
exponential_avg_factor%
О#<ъ
$batch_normalization_2/AssignNewValueAssignVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(е
&batch_normalization_2/AssignNewValue_1AssignVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(r
ReluRelu*batch_normalization_2/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:         @i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:         @Џ
NoOpNoOp%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1 ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':         >> : : : : : : 2L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp:W S
/
_output_shapes
:         >> 
 
_user_specified_nameinputs"х	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*х
serving_defaultА
E
input_1:
serving_default_input_1:0         ђђ<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:йє
ц
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
convolutional_portion
	average_pooling

dense_portion
	optimizer

signatures"
_tf_keras_model
Ш
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21
#22
$23
%24
&25
'26
(27"
trackable_list_wrapper
Х
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
%16
&17
'18
(19"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
)non_trainable_variables

*layers
+metrics
,layer_regularization_losses
-layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ђ
.trace_0
/trace_1
0trace_2
1trace_32ќ
3__inference_convolutional_model_layer_call_fn_66344
3__inference_convolutional_model_layer_call_fn_66846
3__inference_convolutional_model_layer_call_fn_66907
3__inference_convolutional_model_layer_call_fn_66590┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z.trace_0z/trace_1z0trace_2z1trace_3
ь
2trace_0
3trace_1
4trace_2
5trace_32ѓ
N__inference_convolutional_model_layer_call_and_return_conditional_losses_67016
N__inference_convolutional_model_layer_call_and_return_conditional_losses_67132
N__inference_convolutional_model_layer_call_and_return_conditional_losses_66653
N__inference_convolutional_model_layer_call_and_return_conditional_losses_66716┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z2trace_0z3trace_1z4trace_2z5trace_3
╦B╚
 __inference__wrapped_model_64814input_1"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
к
6layer_with_weights-0
6layer-0
7layer_with_weights-1
7layer-1
8layer_with_weights-2
8layer-2
9layer_with_weights-3
9layer-3
:	variables
;trainable_variables
<regularization_losses
=	keras_api
>__call__
*?&call_and_return_all_conditional_losses"
_tf_keras_sequential
Ц
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
Ё
Flayer_with_weights-0
Flayer-0
Glayer-1
Hlayer_with_weights-1
Hlayer-2
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses"
_tf_keras_sequential
с
Oiter

Pbeta_1

Qbeta_2
	Rdecay
Slearning_ratemБmцmЦmдmДmеmЕmфmФmгmГm«m» m░!m▒"m▓%m│&m┤'mх(mХvиvИv╣v║v╗v╝vйvЙv┐v└v┴v┬v├ v─!v┼"vк%vК&v╚'v╔(v╩"
	optimizer
,
Tserving_default"
signature_map
;:9 2!convolutional_block/conv2d/kernel
-:+ 2convolutional_block/conv2d/bias
;:9 2-convolutional_block/batch_normalization/gamma
::8 2,convolutional_block/batch_normalization/beta
C:A  (23convolutional_block/batch_normalization/moving_mean
G:E  (27convolutional_block/batch_normalization/moving_variance
?:=  2%convolutional_block_1/conv2d_1/kernel
1:/ 2#convolutional_block_1/conv2d_1/bias
?:= 21convolutional_block_1/batch_normalization_1/gamma
>:< 20convolutional_block_1/batch_normalization_1/beta
G:E  (27convolutional_block_1/batch_normalization_1/moving_mean
K:I  (2;convolutional_block_1/batch_normalization_1/moving_variance
?:= @2%convolutional_block_2/conv2d_2/kernel
1:/@2#convolutional_block_2/conv2d_2/bias
?:=@21convolutional_block_2/batch_normalization_2/gamma
>:<@20convolutional_block_2/batch_normalization_2/beta
G:E@ (27convolutional_block_2/batch_normalization_2/moving_mean
K:I@ (2;convolutional_block_2/batch_normalization_2/moving_variance
?:=@@2%convolutional_block_3/conv2d_3/kernel
1:/@2#convolutional_block_3/conv2d_3/bias
?:=@21convolutional_block_3/batch_normalization_3/gamma
>:<@20convolutional_block_3/batch_normalization_3/beta
G:E@ (27convolutional_block_3/batch_normalization_3/moving_mean
K:I@ (2;convolutional_block_3/batch_normalization_3/moving_variance
:	@ђ2dense/kernel
:ђ2
dense/bias
!:	ђ2dense_1/kernel
:2dense_1/bias
X
0
1
2
3
4
5
#6
$7"
trackable_list_wrapper
5
0
	1

2"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЁBѓ
3__inference_convolutional_model_layer_call_fn_66344input_1"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ёBЂ
3__inference_convolutional_model_layer_call_fn_66846inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ёBЂ
3__inference_convolutional_model_layer_call_fn_66907inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЁBѓ
3__inference_convolutional_model_layer_call_fn_66590input_1"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЪBю
N__inference_convolutional_model_layer_call_and_return_conditional_losses_67016inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЪBю
N__inference_convolutional_model_layer_call_and_return_conditional_losses_67132inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
аBЮ
N__inference_convolutional_model_layer_call_and_return_conditional_losses_66653input_1"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
аBЮ
N__inference_convolutional_model_layer_call_and_return_conditional_losses_66716input_1"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▄
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]conv_2d
^max_pool_2d
_batch_normalization"
_tf_keras_model
▄
`	variables
atrainable_variables
bregularization_losses
c	keras_api
d__call__
*e&call_and_return_all_conditional_losses
fconv_2d
gmax_pool_2d
hbatch_normalization"
_tf_keras_model
▄
i	variables
jtrainable_variables
kregularization_losses
l	keras_api
m__call__
*n&call_and_return_all_conditional_losses
oconv_2d
pmax_pool_2d
qbatch_normalization"
_tf_keras_model
▄
r	variables
strainable_variables
tregularization_losses
u	keras_api
v__call__
*w&call_and_return_all_conditional_losses
xconv_2d
ymax_pool_2d
zbatch_normalization"
_tf_keras_model
о
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
 19
!20
"21
#22
$23"
trackable_list_wrapper
ќ
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
{non_trainable_variables

|layers
}metrics
~layer_regularization_losses
layer_metrics
:	variables
;trainable_variables
<regularization_losses
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
т
ђtrace_0
Ђtrace_1
ѓtrace_2
Ѓtrace_32Ы
*__inference_sequential_layer_call_fn_65691
*__inference_sequential_layer_call_fn_67185
*__inference_sequential_layer_call_fn_67238
*__inference_sequential_layer_call_fn_65906┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zђtrace_0zЂtrace_1zѓtrace_2zЃtrace_3
Л
ёtrace_0
Ёtrace_1
єtrace_2
Єtrace_32я
E__inference_sequential_layer_call_and_return_conditional_losses_67330
E__inference_sequential_layer_call_and_return_conditional_losses_67422
E__inference_sequential_layer_call_and_return_conditional_losses_65962
E__inference_sequential_layer_call_and_return_conditional_losses_66018┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zёtrace_0zЁtrace_1zєtrace_2zЄtrace_3
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ѕnon_trainable_variables
Ѕlayers
іmetrics
 Іlayer_regularization_losses
їlayer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
■
Їtrace_02▀
8__inference_global_average_pooling2d_layer_call_fn_67427б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЇtrace_0
Ў
јtrace_02Щ
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_67433б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zјtrace_0
┴
Ј	variables
љtrainable_variables
Љregularization_losses
њ	keras_api
Њ__call__
+ћ&call_and_return_all_conditional_losses

%kernel
&bias"
_tf_keras_layer
├
Ћ	variables
ќtrainable_variables
Ќregularization_losses
ў	keras_api
Ў__call__
+џ&call_and_return_all_conditional_losses
Џ_random_generator"
_tf_keras_layer
┴
ю	variables
Юtrainable_variables
ъregularization_losses
Ъ	keras_api
а__call__
+А&call_and_return_all_conditional_losses

'kernel
(bias"
_tf_keras_layer
<
%0
&1
'2
(3"
trackable_list_wrapper
<
%0
&1
'2
(3"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
бnon_trainable_variables
Бlayers
цmetrics
 Цlayer_regularization_losses
дlayer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
ь
Дtrace_0
еtrace_1
Еtrace_2
фtrace_32Щ
,__inference_sequential_1_layer_call_fn_66091
,__inference_sequential_1_layer_call_fn_67446
,__inference_sequential_1_layer_call_fn_67459
,__inference_sequential_1_layer_call_fn_66188┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zДtrace_0zеtrace_1zЕtrace_2zфtrace_3
┘
Фtrace_0
гtrace_1
Гtrace_2
«trace_32Т
G__inference_sequential_1_layer_call_and_return_conditional_losses_67478
G__inference_sequential_1_layer_call_and_return_conditional_losses_67504
G__inference_sequential_1_layer_call_and_return_conditional_losses_66203
G__inference_sequential_1_layer_call_and_return_conditional_losses_66218┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zФtrace_0zгtrace_1zГtrace_2z«trace_3
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
╩BК
#__inference_signature_wrapper_66785input_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
R
»	variables
░	keras_api

▒total

▓count"
_tf_keras_metric
c
│	variables
┤	keras_api

хtotal

Хcount
и
_fn_kwargs"
_tf_keras_metric
J
0
1
2
3
4
5"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Иnon_trainable_variables
╣layers
║metrics
 ╗layer_regularization_losses
╝layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
Ѕ
йtrace_0
Йtrace_1
┐trace_2
└trace_32ќ
3__inference_convolutional_block_layer_call_fn_64940
3__inference_convolutional_block_layer_call_fn_67521
3__inference_convolutional_block_layer_call_fn_67538
3__inference_convolutional_block_layer_call_fn_65021┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zйtrace_0zЙtrace_1z┐trace_2z└trace_3
ш
┴trace_0
┬trace_1
├trace_2
─trace_32ѓ
N__inference_convolutional_block_layer_call_and_return_conditional_losses_67564
N__inference_convolutional_block_layer_call_and_return_conditional_losses_67590
N__inference_convolutional_block_layer_call_and_return_conditional_losses_65041
N__inference_convolutional_block_layer_call_and_return_conditional_losses_65061┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┴trace_0z┬trace_1z├trace_2z─trace_3
С
┼	variables
кtrainable_variables
Кregularization_losses
╚	keras_api
╔__call__
+╩&call_and_return_all_conditional_losses

kernel
bias
!╦_jit_compiled_convolution_op"
_tf_keras_layer
Ф
╠	variables
═trainable_variables
╬regularization_losses
¤	keras_api
л__call__
+Л&call_and_return_all_conditional_losses"
_tf_keras_layer
ы
м	variables
Мtrainable_variables
нregularization_losses
Н	keras_api
о__call__
+О&call_and_return_all_conditional_losses
	пaxis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
┘non_trainable_variables
┌layers
█metrics
 ▄layer_regularization_losses
Пlayer_metrics
`	variables
atrainable_variables
bregularization_losses
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
в
яtrace_0
▀trace_12░
5__inference_convolutional_block_1_layer_call_fn_67607
5__inference_convolutional_block_1_layer_call_fn_67624┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zяtrace_0z▀trace_1
А
Яtrace_0
рtrace_12Т
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_67650
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_67676┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЯtrace_0zрtrace_1
С
Р	variables
сtrainable_variables
Сregularization_losses
т	keras_api
Т__call__
+у&call_and_return_all_conditional_losses

kernel
bias
!У_jit_compiled_convolution_op"
_tf_keras_layer
Ф
ж	variables
Жtrainable_variables
вregularization_losses
В	keras_api
ь__call__
+Ь&call_and_return_all_conditional_losses"
_tf_keras_layer
ы
№	variables
­trainable_variables
ыregularization_losses
Ы	keras_api
з__call__
+З&call_and_return_all_conditional_losses
	шaxis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
J
0
1
2
3
4
5"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Шnon_trainable_variables
эlayers
Эmetrics
 щlayer_regularization_losses
Щlayer_metrics
i	variables
jtrainable_variables
kregularization_losses
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
в
чtrace_0
Чtrace_12░
5__inference_convolutional_block_2_layer_call_fn_67693
5__inference_convolutional_block_2_layer_call_fn_67710┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zчtrace_0zЧtrace_1
А
§trace_0
■trace_12Т
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_67736
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_67762┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z§trace_0z■trace_1
С
 	variables
ђtrainable_variables
Ђregularization_losses
ѓ	keras_api
Ѓ__call__
+ё&call_and_return_all_conditional_losses

kernel
bias
!Ё_jit_compiled_convolution_op"
_tf_keras_layer
Ф
є	variables
Єtrainable_variables
ѕregularization_losses
Ѕ	keras_api
і__call__
+І&call_and_return_all_conditional_losses"
_tf_keras_layer
ы
ї	variables
Їtrainable_variables
јregularization_losses
Ј	keras_api
љ__call__
+Љ&call_and_return_all_conditional_losses
	њaxis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
J
0
 1
!2
"3
#4
$5"
trackable_list_wrapper
<
0
 1
!2
"3"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Њnon_trainable_variables
ћlayers
Ћmetrics
 ќlayer_regularization_losses
Ќlayer_metrics
r	variables
strainable_variables
tregularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
в
ўtrace_0
Ўtrace_12░
5__inference_convolutional_block_3_layer_call_fn_67779
5__inference_convolutional_block_3_layer_call_fn_67796┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zўtrace_0zЎtrace_1
А
џtrace_0
Џtrace_12Т
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_67822
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_67848┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zџtrace_0zЏtrace_1
С
ю	variables
Юtrainable_variables
ъregularization_losses
Ъ	keras_api
а__call__
+А&call_and_return_all_conditional_losses

kernel
 bias
!б_jit_compiled_convolution_op"
_tf_keras_layer
Ф
Б	variables
цtrainable_variables
Цregularization_losses
д	keras_api
Д__call__
+е&call_and_return_all_conditional_losses"
_tf_keras_layer
ы
Е	variables
фtrainable_variables
Фregularization_losses
г	keras_api
Г__call__
+«&call_and_return_all_conditional_losses
	»axis
	!gamma
"beta
#moving_mean
$moving_variance"
_tf_keras_layer
X
0
1
2
3
4
5
#6
$7"
trackable_list_wrapper
<
60
71
82
93"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
јBІ
*__inference_sequential_layer_call_fn_65691convolutional_block_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
чBЭ
*__inference_sequential_layer_call_fn_67185inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
чBЭ
*__inference_sequential_layer_call_fn_67238inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
јBІ
*__inference_sequential_layer_call_fn_65906convolutional_block_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
E__inference_sequential_layer_call_and_return_conditional_losses_67330inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ќBЊ
E__inference_sequential_layer_call_and_return_conditional_losses_67422inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЕBд
E__inference_sequential_layer_call_and_return_conditional_losses_65962convolutional_block_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЕBд
E__inference_sequential_layer_call_and_return_conditional_losses_66018convolutional_block_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
ВBж
8__inference_global_average_pooling2d_layer_call_fn_67427inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЄBё
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_67433inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
░non_trainable_variables
▒layers
▓metrics
 │layer_regularization_losses
┤layer_metrics
Ј	variables
љtrainable_variables
Љregularization_losses
Њ__call__
+ћ&call_and_return_all_conditional_losses
'ћ"call_and_return_conditional_losses"
_generic_user_object
в
хtrace_02╠
%__inference_dense_layer_call_fn_67857б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zхtrace_0
є
Хtrace_02у
@__inference_dense_layer_call_and_return_conditional_losses_67868б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zХtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
иnon_trainable_variables
Иlayers
╣metrics
 ║layer_regularization_losses
╗layer_metrics
Ћ	variables
ќtrainable_variables
Ќregularization_losses
Ў__call__
+џ&call_and_return_all_conditional_losses
'џ"call_and_return_conditional_losses"
_generic_user_object
├
╝trace_0
йtrace_12ѕ
'__inference_dropout_layer_call_fn_67873
'__inference_dropout_layer_call_fn_67878│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╝trace_0zйtrace_1
щ
Йtrace_0
┐trace_12Й
B__inference_dropout_layer_call_and_return_conditional_losses_67883
B__inference_dropout_layer_call_and_return_conditional_losses_67895│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЙtrace_0z┐trace_1
"
_generic_user_object
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
ю	variables
Юtrainable_variables
ъregularization_losses
а__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
ь
┼trace_02╬
'__inference_dense_1_layer_call_fn_67904б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┼trace_0
ѕ
кtrace_02ж
B__inference_dense_1_layer_call_and_return_conditional_losses_67915б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zкtrace_0
 "
trackable_list_wrapper
5
F0
G1
H2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ѓB 
,__inference_sequential_1_layer_call_fn_66091dense_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
§BЩ
,__inference_sequential_1_layer_call_fn_67446inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
§BЩ
,__inference_sequential_1_layer_call_fn_67459inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ѓB 
,__inference_sequential_1_layer_call_fn_66188dense_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ўBЋ
G__inference_sequential_1_layer_call_and_return_conditional_losses_67478inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ўBЋ
G__inference_sequential_1_layer_call_and_return_conditional_losses_67504inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЮBџ
G__inference_sequential_1_layer_call_and_return_conditional_losses_66203dense_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЮBџ
G__inference_sequential_1_layer_call_and_return_conditional_losses_66218dense_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
0
▒0
▓1"
trackable_list_wrapper
.
»	variables"
_generic_user_object
:  (2total
:  (2count
0
х0
Х1"
trackable_list_wrapper
.
│	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
5
]0
^1
_2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЌBћ
3__inference_convolutional_block_layer_call_fn_64940convolutional_block_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ёBЂ
3__inference_convolutional_block_layer_call_fn_67521inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ёBЂ
3__inference_convolutional_block_layer_call_fn_67538inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЌBћ
3__inference_convolutional_block_layer_call_fn_65021convolutional_block_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЪBю
N__inference_convolutional_block_layer_call_and_return_conditional_losses_67564inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЪBю
N__inference_convolutional_block_layer_call_and_return_conditional_losses_67590inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▓B»
N__inference_convolutional_block_layer_call_and_return_conditional_losses_65041convolutional_block_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
▓B»
N__inference_convolutional_block_layer_call_and_return_conditional_losses_65061convolutional_block_input"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Кnon_trainable_variables
╚layers
╔metrics
 ╩layer_regularization_losses
╦layer_metrics
┼	variables
кtrainable_variables
Кregularization_losses
╔__call__
+╩&call_and_return_all_conditional_losses
'╩"call_and_return_conditional_losses"
_generic_user_object
В
╠trace_02═
&__inference_conv2d_layer_call_fn_67924б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z╠trace_0
Є
═trace_02У
A__inference_conv2d_layer_call_and_return_conditional_losses_67934б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z═trace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
╬non_trainable_variables
¤layers
лmetrics
 Лlayer_regularization_losses
мlayer_metrics
╠	variables
═trainable_variables
╬regularization_losses
л__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
з
Мtrace_02н
-__inference_max_pooling2d_layer_call_fn_67939б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zМtrace_0
ј
нtrace_02№
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_67944б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zнtrace_0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Нnon_trainable_variables
оlayers
Оmetrics
 пlayer_regularization_losses
┘layer_metrics
м	variables
Мtrainable_variables
нregularization_losses
о__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
█
┌trace_0
█trace_12а
3__inference_batch_normalization_layer_call_fn_67957
3__inference_batch_normalization_layer_call_fn_67970│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z┌trace_0z█trace_1
Љ
▄trace_0
Пtrace_12о
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67988
N__inference_batch_normalization_layer_call_and_return_conditional_losses_68006│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 z▄trace_0zПtrace_1
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
5
f0
g1
h2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
єBЃ
5__inference_convolutional_block_1_layer_call_fn_67607inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
єBЃ
5__inference_convolutional_block_1_layer_call_fn_67624inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
АBъ
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_67650inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
АBъ
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_67676inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
яnon_trainable_variables
▀layers
Яmetrics
 рlayer_regularization_losses
Рlayer_metrics
Р	variables
сtrainable_variables
Сregularization_losses
Т__call__
+у&call_and_return_all_conditional_losses
'у"call_and_return_conditional_losses"
_generic_user_object
Ь
сtrace_02¤
(__inference_conv2d_1_layer_call_fn_68015б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zсtrace_0
Ѕ
Сtrace_02Ж
C__inference_conv2d_1_layer_call_and_return_conditional_losses_68025б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zСtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
тnon_trainable_variables
Тlayers
уmetrics
 Уlayer_regularization_losses
жlayer_metrics
ж	variables
Жtrainable_variables
вregularization_losses
ь__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
ш
Жtrace_02о
/__inference_max_pooling2d_1_layer_call_fn_68030б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЖtrace_0
љ
вtrace_02ы
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_68035б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zвtrace_0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Вnon_trainable_variables
ьlayers
Ьmetrics
 №layer_regularization_losses
­layer_metrics
№	variables
­trainable_variables
ыregularization_losses
з__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
▀
ыtrace_0
Ыtrace_12ц
5__inference_batch_normalization_1_layer_call_fn_68048
5__inference_batch_normalization_1_layer_call_fn_68061│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zыtrace_0zЫtrace_1
Ћ
зtrace_0
Зtrace_12┌
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68079
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68097│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zзtrace_0zЗtrace_1
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
5
o0
p1
q2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
єBЃ
5__inference_convolutional_block_2_layer_call_fn_67693inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
єBЃ
5__inference_convolutional_block_2_layer_call_fn_67710inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
АBъ
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_67736inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
АBъ
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_67762inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
шnon_trainable_variables
Шlayers
эmetrics
 Эlayer_regularization_losses
щlayer_metrics
 	variables
ђtrainable_variables
Ђregularization_losses
Ѓ__call__
+ё&call_and_return_all_conditional_losses
'ё"call_and_return_conditional_losses"
_generic_user_object
Ь
Щtrace_02¤
(__inference_conv2d_2_layer_call_fn_68106б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЩtrace_0
Ѕ
чtrace_02Ж
C__inference_conv2d_2_layer_call_and_return_conditional_losses_68116б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zчtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Чnon_trainable_variables
§layers
■metrics
  layer_regularization_losses
ђlayer_metrics
є	variables
Єtrainable_variables
ѕregularization_losses
і__call__
+І&call_and_return_all_conditional_losses
'І"call_and_return_conditional_losses"
_generic_user_object
ш
Ђtrace_02о
/__inference_max_pooling2d_2_layer_call_fn_68121б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЂtrace_0
љ
ѓtrace_02ы
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_68126б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zѓtrace_0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ѓnon_trainable_variables
ёlayers
Ёmetrics
 єlayer_regularization_losses
Єlayer_metrics
ї	variables
Їtrainable_variables
јregularization_losses
љ__call__
+Љ&call_and_return_all_conditional_losses
'Љ"call_and_return_conditional_losses"
_generic_user_object
▀
ѕtrace_0
Ѕtrace_12ц
5__inference_batch_normalization_2_layer_call_fn_68139
5__inference_batch_normalization_2_layer_call_fn_68152│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zѕtrace_0zЅtrace_1
Ћ
іtrace_0
Іtrace_12┌
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68170
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68188│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zіtrace_0zІtrace_1
 "
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
5
x0
y1
z2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
єBЃ
5__inference_convolutional_block_3_layer_call_fn_67779inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
єBЃ
5__inference_convolutional_block_3_layer_call_fn_67796inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
АBъ
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_67822inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
АBъ
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_67848inputs"┐
Х▓▓
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsб
p 

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
їnon_trainable_variables
Їlayers
јmetrics
 Јlayer_regularization_losses
љlayer_metrics
ю	variables
Юtrainable_variables
ъregularization_losses
а__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
Ь
Љtrace_02¤
(__inference_conv2d_3_layer_call_fn_68197б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЉtrace_0
Ѕ
њtrace_02Ж
C__inference_conv2d_3_layer_call_and_return_conditional_losses_68207б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zњtrace_0
┤2▒«
Б▓Ъ
FullArgSpec'
argsџ
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Њnon_trainable_variables
ћlayers
Ћmetrics
 ќlayer_regularization_losses
Ќlayer_metrics
Б	variables
цtrainable_variables
Цregularization_losses
Д__call__
+е&call_and_return_all_conditional_losses
'е"call_and_return_conditional_losses"
_generic_user_object
ш
ўtrace_02о
/__inference_max_pooling2d_3_layer_call_fn_68212б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zўtrace_0
љ
Ўtrace_02ы
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_68217б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЎtrace_0
<
!0
"1
#2
$3"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
џnon_trainable_variables
Џlayers
юmetrics
 Юlayer_regularization_losses
ъlayer_metrics
Е	variables
фtrainable_variables
Фregularization_losses
Г__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
▀
Ъtrace_0
аtrace_12ц
5__inference_batch_normalization_3_layer_call_fn_68230
5__inference_batch_normalization_3_layer_call_fn_68243│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zЪtrace_0zаtrace_1
Ћ
Аtrace_0
бtrace_12┌
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68261
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68279│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 zАtrace_0zбtrace_1
 "
trackable_list_wrapper
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
┘Bо
%__inference_dense_layer_call_fn_67857inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЗBы
@__inference_dense_layer_call_and_return_conditional_losses_67868inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
ВBж
'__inference_dropout_layer_call_fn_67873inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ВBж
'__inference_dropout_layer_call_fn_67878inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЄBё
B__inference_dropout_layer_call_and_return_conditional_losses_67883inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЄBё
B__inference_dropout_layer_call_and_return_conditional_losses_67895inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
█Bп
'__inference_dense_1_layer_call_fn_67904inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ШBз
B__inference_dense_1_layer_call_and_return_conditional_losses_67915inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
┌BО
&__inference_conv2d_layer_call_fn_67924inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
шBЫ
A__inference_conv2d_layer_call_and_return_conditional_losses_67934inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
рBя
-__inference_max_pooling2d_layer_call_fn_67939inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЧBщ
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_67944inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЭBш
3__inference_batch_normalization_layer_call_fn_67957inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЭBш
3__inference_batch_normalization_layer_call_fn_67970inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЊBљ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67988inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЊBљ
N__inference_batch_normalization_layer_call_and_return_conditional_losses_68006inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
▄B┘
(__inference_conv2d_1_layer_call_fn_68015inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
C__inference_conv2d_1_layer_call_and_return_conditional_losses_68025inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
сBЯ
/__inference_max_pooling2d_1_layer_call_fn_68030inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_68035inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЩBэ
5__inference_batch_normalization_1_layer_call_fn_68048inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
5__inference_batch_normalization_1_layer_call_fn_68061inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68079inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68097inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
▄B┘
(__inference_conv2d_2_layer_call_fn_68106inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
C__inference_conv2d_2_layer_call_and_return_conditional_losses_68116inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
сBЯ
/__inference_max_pooling2d_2_layer_call_fn_68121inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_68126inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЩBэ
5__inference_batch_normalization_2_layer_call_fn_68139inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
5__inference_batch_normalization_2_layer_call_fn_68152inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68170inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68188inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
▄B┘
(__inference_conv2d_3_layer_call_fn_68197inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
эBЗ
C__inference_conv2d_3_layer_call_and_return_conditional_losses_68207inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
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
сBЯ
/__inference_max_pooling2d_3_layer_call_fn_68212inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
■Bч
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_68217inputs"б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЩBэ
5__inference_batch_normalization_3_layer_call_fn_68230inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЩBэ
5__inference_batch_normalization_3_layer_call_fn_68243inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68261inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ЋBњ
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68279inputs"│
ф▓д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
@:> 2(Adam/convolutional_block/conv2d/kernel/m
2:0 2&Adam/convolutional_block/conv2d/bias/m
@:> 24Adam/convolutional_block/batch_normalization/gamma/m
?:= 23Adam/convolutional_block/batch_normalization/beta/m
D:B  2,Adam/convolutional_block_1/conv2d_1/kernel/m
6:4 2*Adam/convolutional_block_1/conv2d_1/bias/m
D:B 28Adam/convolutional_block_1/batch_normalization_1/gamma/m
C:A 27Adam/convolutional_block_1/batch_normalization_1/beta/m
D:B @2,Adam/convolutional_block_2/conv2d_2/kernel/m
6:4@2*Adam/convolutional_block_2/conv2d_2/bias/m
D:B@28Adam/convolutional_block_2/batch_normalization_2/gamma/m
C:A@27Adam/convolutional_block_2/batch_normalization_2/beta/m
D:B@@2,Adam/convolutional_block_3/conv2d_3/kernel/m
6:4@2*Adam/convolutional_block_3/conv2d_3/bias/m
D:B@28Adam/convolutional_block_3/batch_normalization_3/gamma/m
C:A@27Adam/convolutional_block_3/batch_normalization_3/beta/m
$:"	@ђ2Adam/dense/kernel/m
:ђ2Adam/dense/bias/m
&:$	ђ2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
@:> 2(Adam/convolutional_block/conv2d/kernel/v
2:0 2&Adam/convolutional_block/conv2d/bias/v
@:> 24Adam/convolutional_block/batch_normalization/gamma/v
?:= 23Adam/convolutional_block/batch_normalization/beta/v
D:B  2,Adam/convolutional_block_1/conv2d_1/kernel/v
6:4 2*Adam/convolutional_block_1/conv2d_1/bias/v
D:B 28Adam/convolutional_block_1/batch_normalization_1/gamma/v
C:A 27Adam/convolutional_block_1/batch_normalization_1/beta/v
D:B @2,Adam/convolutional_block_2/conv2d_2/kernel/v
6:4@2*Adam/convolutional_block_2/conv2d_2/bias/v
D:B@28Adam/convolutional_block_2/batch_normalization_2/gamma/v
C:A@27Adam/convolutional_block_2/batch_normalization_2/beta/v
D:B@@2,Adam/convolutional_block_3/conv2d_3/kernel/v
6:4@2*Adam/convolutional_block_3/conv2d_3/bias/v
D:B@28Adam/convolutional_block_3/batch_normalization_3/gamma/v
C:A@27Adam/convolutional_block_3/batch_normalization_3/beta/v
$:"	@ђ2Adam/dense/kernel/v
:ђ2Adam/dense/bias/v
&:$	ђ2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v┤
 __inference__wrapped_model_64814Ј !"#$%&'(:б7
0б-
+і(
input_1         ђђ
ф "3ф0
.
output_1"і
output_1         в
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68079ќMбJ
Cб@
:і7
inputs+                            
p 
ф "?б<
5і2
0+                            
џ в
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_68097ќMбJ
Cб@
:і7
inputs+                            
p
ф "?б<
5і2
0+                            
џ ├
5__inference_batch_normalization_1_layer_call_fn_68048ЅMбJ
Cб@
:і7
inputs+                            
p 
ф "2і/+                            ├
5__inference_batch_normalization_1_layer_call_fn_68061ЅMбJ
Cб@
:і7
inputs+                            
p
ф "2і/+                            в
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68170ќMбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ в
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_68188ќMбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ ├
5__inference_batch_normalization_2_layer_call_fn_68139ЅMбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @├
5__inference_batch_normalization_2_layer_call_fn_68152ЅMбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @в
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68261ќ!"#$MбJ
Cб@
:і7
inputs+                           @
p 
ф "?б<
5і2
0+                           @
џ в
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_68279ќ!"#$MбJ
Cб@
:і7
inputs+                           @
p
ф "?б<
5і2
0+                           @
џ ├
5__inference_batch_normalization_3_layer_call_fn_68230Ѕ!"#$MбJ
Cб@
:і7
inputs+                           @
p 
ф "2і/+                           @├
5__inference_batch_normalization_3_layer_call_fn_68243Ѕ!"#$MбJ
Cб@
:і7
inputs+                           @
p
ф "2і/+                           @ж
N__inference_batch_normalization_layer_call_and_return_conditional_losses_67988ќMбJ
Cб@
:і7
inputs+                            
p 
ф "?б<
5і2
0+                            
џ ж
N__inference_batch_normalization_layer_call_and_return_conditional_losses_68006ќMбJ
Cб@
:і7
inputs+                            
p
ф "?б<
5і2
0+                            
џ ┴
3__inference_batch_normalization_layer_call_fn_67957ЅMбJ
Cб@
:і7
inputs+                            
p 
ф "2і/+                            ┴
3__inference_batch_normalization_layer_call_fn_67970ЅMбJ
Cб@
:і7
inputs+                            
p
ф "2і/+                            │
C__inference_conv2d_1_layer_call_and_return_conditional_losses_68025l7б4
-б*
(і%
inputs          
ф "-б*
#і 
0         }} 
џ І
(__inference_conv2d_1_layer_call_fn_68015_7б4
-б*
(і%
inputs          
ф " і         }} │
C__inference_conv2d_2_layer_call_and_return_conditional_losses_68116l7б4
-б*
(і%
inputs         >> 
ф "-б*
#і 
0         <<@
џ І
(__inference_conv2d_2_layer_call_fn_68106_7б4
-б*
(і%
inputs         >> 
ф " і         <<@│
C__inference_conv2d_3_layer_call_and_return_conditional_losses_68207l 7б4
-б*
(і%
inputs         @
ф "-б*
#і 
0         @
џ І
(__inference_conv2d_3_layer_call_fn_68197_ 7б4
-б*
(і%
inputs         @
ф " і         @х
A__inference_conv2d_layer_call_and_return_conditional_losses_67934p9б6
/б,
*і'
inputs         ђђ
ф "/б,
%і"
0         ■■ 
џ Ї
&__inference_conv2d_layer_call_fn_67924c9б6
/б,
*і'
inputs         ђђ
ф ""і         ■■ ╠
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_67650x?б<
5б2
(і%
inputs          
p 

 
ф "-б*
#і 
0         >> 
џ ╠
P__inference_convolutional_block_1_layer_call_and_return_conditional_losses_67676x?б<
5б2
(і%
inputs          
p

 
ф "-б*
#і 
0         >> 
џ ц
5__inference_convolutional_block_1_layer_call_fn_67607k?б<
5б2
(і%
inputs          
p 

 
ф " і         >> ц
5__inference_convolutional_block_1_layer_call_fn_67624k?б<
5б2
(і%
inputs          
p

 
ф " і         >> ╠
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_67736x?б<
5б2
(і%
inputs         >> 
p 

 
ф "-б*
#і 
0         @
џ ╠
P__inference_convolutional_block_2_layer_call_and_return_conditional_losses_67762x?б<
5б2
(і%
inputs         >> 
p

 
ф "-б*
#і 
0         @
џ ц
5__inference_convolutional_block_2_layer_call_fn_67693k?б<
5б2
(і%
inputs         >> 
p 

 
ф " і         @ц
5__inference_convolutional_block_2_layer_call_fn_67710k?б<
5б2
(і%
inputs         >> 
p

 
ф " і         @╠
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_67822x !"#$?б<
5б2
(і%
inputs         @
p 

 
ф "-б*
#і 
0         @
џ ╠
P__inference_convolutional_block_3_layer_call_and_return_conditional_losses_67848x !"#$?б<
5б2
(і%
inputs         @
p

 
ф "-б*
#і 
0         @
џ ц
5__inference_convolutional_block_3_layer_call_fn_67779k !"#$?б<
5б2
(і%
inputs         @
p 

 
ф " і         @ц
5__inference_convolutional_block_3_layer_call_fn_67796k !"#$?б<
5б2
(і%
inputs         @
p

 
ф " і         @Я
N__inference_convolutional_block_layer_call_and_return_conditional_losses_65041ЇTбQ
JбG
=і:
convolutional_block_input         ђђ
p 

 
ф "-б*
#і 
0          
џ Я
N__inference_convolutional_block_layer_call_and_return_conditional_losses_65061ЇTбQ
JбG
=і:
convolutional_block_input         ђђ
p

 
ф "-б*
#і 
0          
џ ╠
N__inference_convolutional_block_layer_call_and_return_conditional_losses_67564zAб>
7б4
*і'
inputs         ђђ
p 

 
ф "-б*
#і 
0          
џ ╠
N__inference_convolutional_block_layer_call_and_return_conditional_losses_67590zAб>
7б4
*і'
inputs         ђђ
p

 
ф "-б*
#і 
0          
џ И
3__inference_convolutional_block_layer_call_fn_64940ђTбQ
JбG
=і:
convolutional_block_input         ђђ
p 

 
ф " і          И
3__inference_convolutional_block_layer_call_fn_65021ђTбQ
JбG
=і:
convolutional_block_input         ђђ
p

 
ф " і          ц
3__inference_convolutional_block_layer_call_fn_67521mAб>
7б4
*і'
inputs         ђђ
p 

 
ф " і          ц
3__inference_convolutional_block_layer_call_fn_67538mAб>
7б4
*і'
inputs         ђђ
p

 
ф " і          ▄
N__inference_convolutional_model_layer_call_and_return_conditional_losses_66653Ѕ !"#$%&'(Bб?
8б5
+і(
input_1         ђђ
p 

 
ф "%б"
і
0         
џ ▄
N__inference_convolutional_model_layer_call_and_return_conditional_losses_66716Ѕ !"#$%&'(Bб?
8б5
+і(
input_1         ђђ
p

 
ф "%б"
і
0         
џ █
N__inference_convolutional_model_layer_call_and_return_conditional_losses_67016ѕ !"#$%&'(Aб>
7б4
*і'
inputs         ђђ
p 

 
ф "%б"
і
0         
џ █
N__inference_convolutional_model_layer_call_and_return_conditional_losses_67132ѕ !"#$%&'(Aб>
7б4
*і'
inputs         ђђ
p

 
ф "%б"
і
0         
џ │
3__inference_convolutional_model_layer_call_fn_66344| !"#$%&'(Bб?
8б5
+і(
input_1         ђђ
p 

 
ф "і         │
3__inference_convolutional_model_layer_call_fn_66590| !"#$%&'(Bб?
8б5
+і(
input_1         ђђ
p

 
ф "і         ▓
3__inference_convolutional_model_layer_call_fn_66846{ !"#$%&'(Aб>
7б4
*і'
inputs         ђђ
p 

 
ф "і         ▓
3__inference_convolutional_model_layer_call_fn_66907{ !"#$%&'(Aб>
7б4
*і'
inputs         ђђ
p

 
ф "і         Б
B__inference_dense_1_layer_call_and_return_conditional_losses_67915]'(0б-
&б#
!і
inputs         ђ
ф "%б"
і
0         
џ {
'__inference_dense_1_layer_call_fn_67904P'(0б-
&б#
!і
inputs         ђ
ф "і         А
@__inference_dense_layer_call_and_return_conditional_losses_67868]%&/б,
%б"
 і
inputs         @
ф "&б#
і
0         ђ
џ y
%__inference_dense_layer_call_fn_67857P%&/б,
%б"
 і
inputs         @
ф "і         ђц
B__inference_dropout_layer_call_and_return_conditional_losses_67883^4б1
*б'
!і
inputs         ђ
p 
ф "&б#
і
0         ђ
џ ц
B__inference_dropout_layer_call_and_return_conditional_losses_67895^4б1
*б'
!і
inputs         ђ
p
ф "&б#
і
0         ђ
џ |
'__inference_dropout_layer_call_fn_67873Q4б1
*б'
!і
inputs         ђ
p 
ф "і         ђ|
'__inference_dropout_layer_call_fn_67878Q4б1
*б'
!і
inputs         ђ
p
ф "і         ђ▄
S__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_67433ёRбO
HбE
Cі@
inputs4                                    
ф ".б+
$і!
0                  
џ │
8__inference_global_average_pooling2d_layer_call_fn_67427wRбO
HбE
Cі@
inputs4                                    
ф "!і                  ь
J__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_68035ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ┼
/__inference_max_pooling2d_1_layer_call_fn_68030ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ь
J__inference_max_pooling2d_2_layer_call_and_return_conditional_losses_68126ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ┼
/__inference_max_pooling2d_2_layer_call_fn_68121ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    ь
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_68217ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ┼
/__inference_max_pooling2d_3_layer_call_fn_68212ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    в
H__inference_max_pooling2d_layer_call_and_return_conditional_losses_67944ъRбO
HбE
Cі@
inputs4                                    
ф "HбE
>і;
04                                    
џ ├
-__inference_max_pooling2d_layer_call_fn_67939ЉRбO
HбE
Cі@
inputs4                                    
ф ";і84                                    Х
G__inference_sequential_1_layer_call_and_return_conditional_losses_66203k%&'(<б9
2б/
%і"
dense_input         @
p 

 
ф "%б"
і
0         
џ Х
G__inference_sequential_1_layer_call_and_return_conditional_losses_66218k%&'(<б9
2б/
%і"
dense_input         @
p

 
ф "%б"
і
0         
џ ▒
G__inference_sequential_1_layer_call_and_return_conditional_losses_67478f%&'(7б4
-б*
 і
inputs         @
p 

 
ф "%б"
і
0         
џ ▒
G__inference_sequential_1_layer_call_and_return_conditional_losses_67504f%&'(7б4
-б*
 і
inputs         @
p

 
ф "%б"
і
0         
џ ј
,__inference_sequential_1_layer_call_fn_66091^%&'(<б9
2б/
%і"
dense_input         @
p 

 
ф "і         ј
,__inference_sequential_1_layer_call_fn_66188^%&'(<б9
2б/
%і"
dense_input         @
p

 
ф "і         Ѕ
,__inference_sequential_1_layer_call_fn_67446Y%&'(7б4
-б*
 і
inputs         @
p 

 
ф "і         Ѕ
,__inference_sequential_1_layer_call_fn_67459Y%&'(7б4
-б*
 і
inputs         @
p

 
ф "і         ж
E__inference_sequential_layer_call_and_return_conditional_losses_65962Ъ !"#$TбQ
JбG
=і:
convolutional_block_input         ђђ
p 

 
ф "-б*
#і 
0         @
џ ж
E__inference_sequential_layer_call_and_return_conditional_losses_66018Ъ !"#$TбQ
JбG
=і:
convolutional_block_input         ђђ
p

 
ф "-б*
#і 
0         @
џ о
E__inference_sequential_layer_call_and_return_conditional_losses_67330ї !"#$Aб>
7б4
*і'
inputs         ђђ
p 

 
ф "-б*
#і 
0         @
џ о
E__inference_sequential_layer_call_and_return_conditional_losses_67422ї !"#$Aб>
7б4
*і'
inputs         ђђ
p

 
ф "-б*
#і 
0         @
џ ┴
*__inference_sequential_layer_call_fn_65691њ !"#$TбQ
JбG
=і:
convolutional_block_input         ђђ
p 

 
ф " і         @┴
*__inference_sequential_layer_call_fn_65906њ !"#$TбQ
JбG
=і:
convolutional_block_input         ђђ
p

 
ф " і         @Г
*__inference_sequential_layer_call_fn_67185 !"#$Aб>
7б4
*і'
inputs         ђђ
p 

 
ф " і         @Г
*__inference_sequential_layer_call_fn_67238 !"#$Aб>
7б4
*і'
inputs         ђђ
p

 
ф " і         @┬
#__inference_signature_wrapper_66785џ !"#$%&'(EбB
б 
;ф8
6
input_1+і(
input_1         ђђ"3ф0
.
output_1"і
output_1         