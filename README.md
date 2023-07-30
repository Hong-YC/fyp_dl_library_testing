# An Automated Testing Tool for Deep Learning Libraries

This is the implementation repository of our Final Year Project: **An Automated Testing Tool for Deep Learning Libraries** supervised by [Professor Shing-Chi Cheung](https://www.cse.ust.hk/~scc/) in HKUST. Our implementation is based on the implementation of [Muffin](https://github.com/library-testing/Muffin).

## Description
Artificial intelligence (AI) has ushered in a new era of technological advancements, but its reliance on deep learning libraries can introduce bugs that are difficult to detect and debug manually.This problem becomes more pronounced when the bugs are not immediately apparent, leading to costly and time-consuming efforts to identify and fix them. To address this challenge, we
developed an automated testing tool for deep learning libraries that uses model-based techniques to detect bugs efficiently. Our experiments demonstrate that our tool successfully identified bugs in the libraries, validating its effectiveness and usefulness in ensuring the reliability of DL libraries. For more detailed information, one can refer to our [report](https://github.com/Hong-YC/fyp_dl_library_testing/blob/master/Report.pdf).

## Environment
To replicate the results shown in the report, install the following packages:
pytorch 1.13.0
tensorflow 2.10.0
onnx 1.12.0
onnx-tf 1.10.0
onnxruntime 1.12.1 
numpy 1.23.4

## Experiment


#### 1. Configuration

A configuration file `testing_config.json` should be provided to flexibly set up testing configuration. Here is an example:

```json
{
    "case_num": 5,
    "data_dir": "data",
    "timeout": 300,
    "use_heuristic": 1
}
```
* `case_num` indicates the number of random models to generate for test.
* `data_dir` indicates the data directory to store outputs.  Remaining `data` is recommended.
* `timeout` indicates the timeout duration for each model.
* `use_heuristic` indicates if using the heuristic method mentioned in the paper or not. `1` is recommended.

#### 2. Generate database
Execute the following command in `/data/data` to create the ***sqlite*** database for storing the testing results:

  ```
  sqlite3 dummy.db < create_db.sql
  ```

#### 3. Start
Use the following command to run the experiment according to the configuration:

```shell
python run.py
```

The `testing_config.json` file should be place in the same directory.

The testing results will be store in `/data/[DATA_DIR]/dummy.db` (*e.g.* `/data/data/dummy.db`), and **detail results** for each model will be stored in `/data/[DATA_DIR]/dummy_output`. 

Use the following command in `/data/data` to delete a set of testing results ( **carefully use!** ):

```shell
python clear_data.py dummy
```



