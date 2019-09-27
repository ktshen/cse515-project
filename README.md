# cse515p2


## Phase 2

#### Task 1 example:
Note: Please run phase 1 task 2 first so that can decrease the time to get image features.
TODO: We need to confirm whether we can use phase task 2 to build database before phase 2 demo.

Using model color moments, load "test" table, topk is 5, and the method of dimension reduction is svd.
```Shell
python p2task1.py -m cm -t test -k 5 -d svd
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
