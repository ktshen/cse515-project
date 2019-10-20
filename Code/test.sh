PYTHON=python3
CODE_PATH=/home/ming/cse515/code
IMAGE_PATH=/home/ming/cse515/Dataset2
META_PATH=/home/ming/cse515/metadata/HandInfo.csv
DATABASE_NAME=test

MODEL="cm lbp hog sift"
DECOMPOSITION="svd pca lda nmf"
K="5 10 20 40"

echo "Creating descriptors"
for model in `echo $MODEL`
do
	$PYTHON $CODE_PATH/p1task2.py \
	-i $IMAGE_PATH \
	-m $model \
	-t $DATABASE_NAME
done

echo "Testing task 1"
for model in `echo $MODEL`
do
	echo "$model"
	for decom in `echo $DECOMPOSITION`
	do
		echo "$decom"
		for k in `echo $K`
		do
		$PYTHON $CODE_PATH/p2task1.py \
		-m $model \
		-t $DATABASE_NAME \
		-k $k \
		-d $decom \
		-p $IMAGE_PATH
		done
	done
done

QUERY_IMAGE=Hand_0000004
DIS_FUN=l1
TOP_K=10

echo "Testing task 2"
for model in `echo $MODEL`
do
	for decom in `echo $DECOMPOSITION`
	do
		for k in `echo $K`
		do
		$PYTHON $CODE_PATH/p2task2.py \
		-m $model \
		-t $DATABASE_NAME \
		-k $k \
		-d $decom \
		-p $IMAGE_PATH \
		-i $QUERY_IMAGE \
		-mi $TOP_K \
		-dis $DIS_FUN 
		done
	done
done

LABEL=n
echo "Testing task 3"
for model in `echo $MODEL`
do
	echo "$model"
	for decom in `echo $DECOMPOSITION`
	do
		echo "$decom"
		for k in `echo $K`
		do
		$PYTHON $CODE_PATH/p2task3.py \
		-m $model \
		-t $DATABASE_NAME \
		-k $k \
		-d $decom \
		-p $IMAGE_PATH \
		-meta $META_PATH \
		-l $LABEL
		done
	done
done

echo "Testing task 4"
for model in `echo $MODEL`
do
	for decom in `echo $DECOMPOSITION`
	do
		for k in `echo $K`
		do
		$PYTHON $CODE_PATH/p2task4.py \
		-m $model \
		-t $DATABASE_NAME \
		-k $k \
		-d $decom \
		-p $IMAGE_PATH \
		-i $QUERY_IMAGE \
		-mi $TOP_K \
		-dis $DIS_FUN \
		-meta $META_PATH \
		-l $LABEL
		done
	done
done
