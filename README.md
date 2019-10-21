# cse515p2

## Phase 2
Note: Please run phase 1 task 2 first to create descriptor database before running any of the following tasks.

Option:

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

-p image dictionary path

-i queried image id

-mi the number of the query results(k)

-meta metadata(HandInfo.csv) path

-l label
- l -> left-hand
- r -> right-hand
- d -> dorsal
- p -> palmar
- a -> accessories
- m -> male
- f -> female

#### Task 1 example:

Using model color moments, load "test" table, topk is 5, image directory is .../img, and the method of dimension reduction is svd.

```Shell
python p2task1.py -m cm -t test -k 5 -d svd -p .../img
```


#### Task 2 example:

Using color moments, load "test" table, topk is 10, the method of dimension reduction is svd, target ID is Hand\_0008110, find m(10) similar images, and the path of dataset is followed by `-p`.

```Shell
python p2task2.py -m cm -t test -k 10 -d svd -i Hand_0008110 -mi 10 -p .../img
```

#### Task 3 example:

Compared to task 1, two argument are added: -meta for metadata path and `-l` for label.

Metadata can be downloaded from the following link. Please download **csv** file.

[https://sites.google.com/view/11khands](https://sites.google.com/view/11khands)

(*.csv) [download](https://drive.google.com/open?id=1RC86-rVOR8c93XAfM9b9R45L7C2B0FdA) (759 KB)

Please use **one character followed** by `-l` to indicate what label you want.

- l -> left-hand
- r -> right-hand
- d -> dorsal
- p -> palmar
- a -> accessories
- m -> male
- f -> female

The following example is to process images which are left-hand.

```Shell
python p2task3.py -m cm -t test -k 5 -p .../img -d svd -meta .../HandInfo.csv -l l
```

#### Task 4 example:

Compared to task 2, two argument are added: -meta for metadata path and `-l` for label.

```Shell
python p2task4.py -m cm -t test -k 5 -d svd -mi 5 -p .../img -i Hand_0008110 -meta .../HandInfo.csv -l l
```

#### Task 5 example:

Task 5 basically needs a labeled image and an unlabled image. To give an unlabeled image, we can assign an image ID in dataset which would not be used in dimension reduction, or an image path that is not belong to dataset.

Using `-i` argument to give an image ID which exists in database.
```Shell
python p2task5.py -m hog -t test -k 5 -d svd -l l -meta ~/hw/cse515_data/HandInfo.csv -i Hand_0000002
```

Using `-ip` argument to give an image path.
```Shell
python p2task5.py -m hog -t test -k 5 -d svd -l l -meta ~/hw/cse515_data/HandInfo.csv -ip ~/hw/cse515_data/Hands/Hand_0010646.jpg
```

#### Task 6 example:

For task 6, we need to specify the subject ID by `-s` argument.
```Shell
python p2task6.py -t test -s 27 -meta HandInfo.csv -p dataset
```

#### Task 7 example
For task 7, number of top k latent semantics.
```Shell
python p2task7.py -t test -k 10 -meta ~/hw/cse515_data/HandInfo.csv
```

#### Task 8 example
For task 8, number of top k latent semantics, the database table name, and the metadata path.
```Shell
python p2task8.py -k 5 -meta ~/hw/cse515_data/HandInfo.csv -t test
```

## Phase 1

#### Task 1 example:
Give input filepath with argument -i and use color moments with argument -m.
```Shell
python p1task1.py -i ~/hw/cse515_data/CSE\ 515\ Fall19\ -\ Smaller\ Dataset/Hand_0008110.jpg -m cm
```

#### Task 2:
Give input filepath with argument -i, use color moments with argument -m, and store to table "test".
```Shell
python p1task2.py -i ~/hw/cse515_data/CSE\ 515\ Fall19\ -\ Smaller\ Dataset -m cm -t test
```

#### Task 3:
Give query filepath with argument -i, use color moments with argument -m, load to "test" table, and topk is 5.
```Shell
python p1task3.py -i ~/hw/cse515_data/CSE\ 515\ Fall19\ -\ Smaller\ Dataset/Hand_0008110.jpg -m cm -t test -k 5
```
