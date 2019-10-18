# cse515p2


## Phase 2

#### Task 1 example:
Note: Please run phase 1 task 2 first so that can decrease the time to get image features.

TODO: We need to confirm whether we can use phase 1 task 2 to build database before phase 2 demo.

Using model color moments, load "test" table, topk is 5, image directory is .../imgPath, and the method of dimension reduction is svd.
```Shell
python p2task1.py -m cm -t test -k 5 -p .../imgPath -d svd
```


#### Task 2 example:
Note: Please run phase 1 task 2 first to build database.

TODO: We may need to find a new way to represent output.

Using color moments, load "test" table, topk is 10, the method of dimension reduction is svd, target ID is Hand\_0008110, find m(10) similar images, and the path of dataset is followed by `-p`.

```Shell
python p2task2.py -m cm -t test -k 10 -d svd -i Hand_0008110 -mi 10 -p ~/hw/cse515_data/CSE\ 515\ Fall19\ -\ Smaller\ Dataset 
```

#### Task 3 example:
Note: Please run phase 1 task 2 first to build database.

TODO: We need to confirm if task 3 is combination of task 1 with label.

In this task, I added two argument: -meta for metadata path and `-l` for label.

Metadata can be downloaded from the following link. Please download **csv** file.

[https://sites.google.com/view/11khands](https://sites.google.com/view/11khands)

(*.csv) [download](https://drive.google.com/open?id=1RC86-rVOR8c93XAfM9b9R45L7C2B0FdA) (759 KB)

According to project specification, we need to process 8 types of label. Please use **one character followed** by `-l` to indicate what label you want.

- l -> left-hand
- r -> right-hand
- d -> dorsal
- p -> palmar
- a -> accessories
- m -> male
- f -> female

The following example is to process images which are left-hand.

```Shell
python p2task3.py -m cm -t test -k 5 -p .../imgPath -d svd -meta ~/hw/cse515_data/HandInfo.csv -l l
```

#### Task 4 example:

Task 4 is very similar to task 2. But we add some arguments to find some images with specific label from metadata.

```Shell
python p2task4.py -m cm -t test -k 5 -d svd -meta ~/hw/cse515_data/HandInfo.csv -l l -mi 5 -p ~/hw/cse515_data/CSE\ 515\ Fall19\ -\ Smaller\ Dataset -i Hand_0008110
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

For task 6, only table name and subject ID are necessary.
```Shell
python p2task6.py -t test -i Hand_0000002
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
