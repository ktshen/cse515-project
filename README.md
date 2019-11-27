# cse515p2

## Phase 3
Note: Please run phase 1 task 2 first to cretea descripter database before running any of the following tasks.

Option:

-c classifier
- svm(Support Vector Machine)
- dtree(Decision Tree)
- ppr(Personalized Page Rank)

-m model:
- cm(color moment)
- lbp(local binary pattern)
- hog(histograms of oriented gradients)
- sift(scale-invariant feature transform)

-t table: The name that has been used when creating descriptor database.

-k the number of latent semantics.

-d decomposition method:
- svd(Singular value decomposition)
- pca(Principal component analysis)
- lda(Latent Dirichlet allocation)
- nmf(Non-negative matrix factorization)

-l labeled image folder path

-u unlabeled image folder path

-meta metadata(.csv) path

-tmeta metadata(.csv) path for unlabeled / test images folder. This is *optional*. If we can provide test label metadata, this task will show the accuracy.

#### Task 4 example:

Some arguments for task4:
-ut The unlabeled table.

-limg Labled raw image path if you do not want to use feature extraction.

-uimg Unlabeled raw image path if you do not want to use feature extraction.

- Load image features and labels for labeled data / unlabeled data.
- Use PCA as dimension reduction with k = 20.

```Shell
python p3task4.py -c svm -m hog -t set1 -d pca -k 20 -ut tSet1 -meta ~/hw/cse515_data/phase3_sample_data/labelled_set1.csv -tmeta ~/hw/cse515_data/phase3_sample_data/unlabelled_set1.csv
```

Load raw image file and labels for labeld data / unlabeled data.
- Use SVD as dimension reduction with k = 100

```Shell
 python p3task4.py -c svm -meta ~/hw/cse515_data/phase3_sample_data/labelled_set1.csv -limg ~/hw/cse515_data/phase3_sample_data/Labelled/Set1 -uimg ~/hw/cse515_data/phase3_sample_data/Unlabelled/Set\ 1/ -tmeta ~/hw/cse515_data/phase3_sample_data/Unlabelled/unlablled_set1.csv -d svd -k 100
```

#### Task 6 example:

**Please run p3task5.py first before p3task6.py.**

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
