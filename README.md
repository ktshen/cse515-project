# cse515p2

## Phase 3
Note: Please run phase 1 task 2 first to cretea descripter database before running any of the following tasks.

### Task 1

Arguments:

-p PATH: The path of labeled images.

-unp PATH: The path of unlabeled images.

-k K: The top k of latent semantics.

-t TABLE: The name of table used to store image features.

-meta PATH: The path of metadata for labeled images.

-test(optional) Test mode: The program will print the accuracy, the meta has to be the complete one.


### Task 2:

Arguments:

-p PATH: The path of labeled images.

-unp PATH: The path of unlabeled images.

-k K: The top k of latent semantics.

-c CLUSTER: The number of clusters.

-t TABLE: The name of table used to store image features.

-meta PATH: The path of metadata for labeled images.

-test(optional) Test mode: The program will print the accuracy, the meta has to be the complete one.

-k(optional) # of latent semantics: The default value is 10. It is usually set in test mode to show the effect of k.
### Task 3:

Arguments:

-k K: The number of outgoing edges.

-lk K: Most K dominant images.

-t TABLE: The name of table used to store image features.

-i PATH: The path of input folder.

-lst ID1,ID2,ID3: The ID of 3 query images.


### Task 4:

Some arguments for task4:

-c CLASSIFIER: The classifier will be used.
- svm(Support Vector Machine)
- dtree(Decision Tree)
- ppr(Personalized Page Rank)

-m MODEL(*optional*):
- cm(color moment)
- lbp(local binary pattern)
- hog(histograms of oriented gradients)
- sift(scale-invariant feature transform)

-t TABLE(*optional*): The name that has been used when creating descriptor database.

-ut TABLE(*optional*): the unlabeled table.

-k K(*optional*) the number of latent semantics.

-d METHOD(*optional*): The method will be used to reduce dimensions.
- svd(Singular value decomposition)
- pca(Principal component analysis)
- lda(Latent Dirichlet allocation)
- nmf(Non-negative matrix factorization)

-l PATH(*optional*): labeled image folder path

-u PATH(*optional*): unlabeled image folder path

-meta PATH(.csv)(*optional*): the path of metadata for labeled images.

-tmeta PATH(.csv)(*optional*): for unlabeled / test images folder. This is *optional*. If we can provide test label metadata, this task will show the accuracy.

-limg PATH(*optional*): Labled raw image path if you do not want to use feature extraction.

-uimg PATH(*optional*): Unlabeled raw image path if you do not want to use feature extraction.

--svm\_pretrained(*optional*): SVM will save its weight to svm/ folder.

--svm\_pretrained PATH(*optional*): SVM will load the weight and will NOT adjust its weight later.

#### Example:
- Load image features and labels for labeled data / unlabeled data.
- Use PCA as dimension reduction with k = 20.

```Shell
python p3task4.py -c svm -m hog -t set1 -d pca -k 20 -ut tSet1 -meta ~/hw/cse515_data/phase3_sample_data/labelled_set1.csv -tmeta ~/hw/cse515_data/phase3_sample_data/unlabelled_set1.csv
```

#### Example:
- Load raw image file and labels for labeld data / unlabeled data.
- Use SVD as dimension reduction with k = 100

```Shell
 python p3task4.py -c svm -meta ~/hw/cse515_data/phase3_sample_data/labelled_set1.csv -limg ~/hw/cse515_data/phase3_sample_data/Labelled/Set1 -uimg ~/hw/cse515_data/phase3_sample_data/Unlabelled/Set\ 1/ -tmeta ~/hw/cse515_data/phase3_sample_data/Unlabelled/unlablled_set1.csv -d svd -k 100
```

### Task 5:

-l LAYERS: The number of layers.

-k HASHES: The number of hashes per layer.

-i ID: The ID of query image.

-t T: The number of most similar images.

-tb TABLE: The name of table used to store image features.

-d METHOD: The method will be used to reduce dimensions.

-dir PATH: The path of images.

--visualize\_vector(*optional*): Visualize the feature vectors.

```Shell
python p3task5.py -l 5 -k 10 -t 20 -i Hand_0000674 -tb 11k -d hog -dir ~/hw/cse515_data/Hands/
```

### Task 6:

**Please run p3task5.py first before p3task6.py.**

-c CLASSIFIER: The classifier will be used.

-m MODEL: The model of image feature will be used.

-t TABLE: The name of table used to store image features.

-d METHOD: The method will be used to reduce dimension.

-k K: The top k latent semantics.


In task6, the there are prompts as user to input r(relevant), i(irrelevant), or ?(do not know).

```Shell
python p3task6.py -c svm -m cavg -d pca -k 20 -t 11k
```



## Phase 1

#### Task 2:
Give input filepath with argument -i, use color moments with argument -m, and store to table "test".
```Shell
python p1task2.py -i ~/hw/cse515_data/CSE\ 515\ Fall19\ -\ Smaller\ Dataset -m cm -t test
```
