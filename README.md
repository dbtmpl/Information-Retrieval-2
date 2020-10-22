# Information-Retrieval-2

Current project structure:
```
IR2/
 - apex/ (Optional)
 - OpenNIR/
 - src/ (this repo)
   - data/
     - qulac/ (here all OpenNIR/anserini stuff will be saved)
     - documents/webclue_docs/
       - {1-200}.json Note: All docs needed now!
```

## Installation and setup

```
# Download and setup the project
git clone https://github.com/d4vidbiertmpl/Information-Retrieval-2.git
mv Information-Retrieval-2/ src/

# Download OpenNIR
git clone https://github.com/Georgetown-IR-Lab/OpenNIR.git

# install the dependencies
pip install -r src/requirements.txt
pip install -e OpenNIR/

```

## Adding and processing the dataset
After the setup is complete, a directory named "webclue_docs" containing {1-200}.json and an empty directory named "webclue_docs_1000" into `/src/data/documents`.

then run
```
cd src/utils
python data_utils.py
```



## Running models

Due to hardcoded paths in the OpenNIR framework, the terminal needs to be located in the OpenNIR directory to run the models
```
# While located in the root directory
cd OpenNIR/
```

### Conv-KNRM

Baseline

```
python ../src/start.py ./config/conv_knrm ../src/config/qulac
```

Query, question and answer implementation using weighted aggregation

```
python ../src/start.py ../src/config/ranker/std_neural_rankers/conv_knrm_qqa ../src/config/qulac
```

Query, question and answer implementation using mean aggregation
```
python ../src/start.py ../src/config/ranker/std_neural_rankers/conv_knrm_qqa_mean_aggr ../src/config/qulac
```

### PACRR

Baseline

```
python ../src/start.py ../src/config/ranker/std_neural_rankers/pacrr_qqa_single_q ../src/config/qulac
```

Query, question and answer implementation using weighted aggregation

```
python ../src/start.py ../src/config/ranker/std_neural_rankers/pacrr_qqa ../src/config/qulac
```

Query, question and answer implementation using mean aggregation
```
python ../src/start.py ../src/config/ranker/std_neural_rankers/pacrr_qqa_mean_aggr ../src/config/qulac
```

### BERT

Baseline

```
python ../src/start.py ./config/vanilla_bert ../src/config/qulac
```

Query, question and answer implementation using Joint BERT

```
python ../src/start.py ../src/config/ranker/bert_rankers/vanilla_bert_qqa_joint_enc ../src/config/qulac
```

Query, question and answer implementation using 1D convolution
```
python ../src/start.py ../src/config/ranker/std_neural_rankers/pacrr_qqa_mean_aggr ../src/config/qulac
```

## Utility functions

All documents are needed now such that the code runs smoothly. Indexing all docs can take up to ~30 minutes. 

Further, the `split_data()` function in `utils/data_utils` creates `train.qrels.txt`, `valid.qrels.txt`, `test.qrels.txt` (Files are also in the git).

Everything else will be generated by running OpenNIR.

We now have our own `ranker`, `trainer` and `vocab` classes located in `src/`. 

Because of some hard coded paths we need to run our stuff from the OpenNIR folder. Like this:
``` 
python ../src/start.py ../src/config/ranker/std_neural_rankers/conv_knrm_qqa ../src/config/qulac
```
By using the OpenNIR`util.Registry` we can put our code (new Dataset, Trainer, Ranker, etc.) in our `src/` for examples.

## Adding and using configs
First add new config to config file e.g. `config/qulac/_dir`. General pattern:
```
<class_key>.<config_name>=<value>

# Possible <class_key> are: 'vocab', 'train_ds', 'ranker', 'trainer', 'valid_ds', 'valid_pred', 'test_ds', 'test_pred', 'pipeline'
# See keys in context dict in start.py
```
Example add embedding aggregation config:
```
vocab=wordvec_hash_qqa
vocab.aggregation=mean
```
Now the config name needs needs a default. So it needs to be set in the `default_config()` function. In the vocab:
```
# The vocab you use e.g. WordvecHashVocabQQA ...
@staticmethod
def default_config():
    result = WordvecVocab.default_config().copy()
    result.update({
        'hashspace': 1000,
        'init_stddev': 0.5,
        'log_miss': False,
        'aggregation': 'mean' <== Add new
    })
    return result
``` 
Now config will be part of the config dict in the `__init__()` of the respective class and can be used:
```
# in WordvecHashVocabQQA class...

def __init__(self, config, logger, random):
    super().__init__(config, logger, random)
    self.enc_aggregation = config['aggregation']

    # Do something with new config
```
If config does not appear most likely the naming of the `<class_key>` (in this case `vocab`) is wrong.

## Evaluation
run gen_split_test_qrel in data_utils.py, the destination should be ./src/data/qulac. This seperatates the input file into "yes", "no", "other" and "idk" qrel files.

To evaluate based on the new qrel files, anserini and sqllite files need to be recomputed, if test has been done before, the corresponding qulac_test_* file needs to be deleted so cached results wont be displayed (see run_test.sh)

The run_test.sh is an example run and the file needs to be moved to the OpenNIR directory to work as intended.

## Troubleshooting 

### Java

The OpenNIR framework relies on Java 11, in some cases the .bashrc file needs to updated to use Java 11 by default, if the a Java error is enctounered after Java 11 has been installed, then add the following to .bashrc
```
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

### Lock

If document processing fails when running a model for the first time, then a docs.sqlite.tmp might be crashing the program. The file is located in `./src/data/qulac/`, deleting the file should solve the problem.

### Other errors

Sometimes fiddling too much with the configuration files can mess up the stored_results directory, one way to fix it is to delete it and regenerate the file from scratch by running a model. This will however delete all the stored results.