#!/usr/bin/env python3
import os.path
import sys
import time
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
# Add package root to pythonpath
sys.path.append(os.path.realpath(f"{__file__}/../../"))
sys.setrecursionlimit(100000)
from cirtorch.datasets.testdataset import configdataset
from cirtorch.utils.evaluate import compute_map_and_print
from asmk import io_helpers, ASMKMethod

FEATURES_URL = "http://ptak.felk.cvut.cz/personal/toliageo/share/how/features/"
DATASETS_URL = "http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/"


def initialize(params, demo_params, globals, logger):
    """Download necessary files and initialize structures"""
    logger.info(f"ECCV20 demo with parameters '{globals['exp_path'].name}'")

    # Download featues
    features = ["%s_%s.pickle" % (x, demo_params['eval_features']) \
                    for x in demo_params['eval_datasets']]
    features.append("%s_%s.pickle" % (demo_params['codebook_dataset'],
                                   demo_params['codebook_features']))
    
    # Download test datasets
    pkls = ["%s/gnd_%s.pickle" % (x, x) for x in demo_params['eval_datasets']]

    # Initialize asmk method wrapper
    return ASMKMethod.initialize_untrained(params)


def train_codebook(asmk, demo_params, globals, logger):
    """The first step of asmk method - training the codebook"""
    codebook_path = f"{globals['exp_path']}/codebook.pkl"
    features_path = f"{globals['root_path']}/features/{demo_params['codebook_dataset']}_" \
                    f"{demo_params['codebook_features']}.pickle"

    print(features_path)
    desc = io_helpers.load_pickle(features_path)

    logger.info(f"Loaded descriptors for codebook")
    asmk = asmk.train_codebook(desc['vecs'], cache_path=codebook_path)

    metadata = asmk.metadata['train_codebook']
    logger.debug(f"Using {metadata['index_class']} index")
    if "load_time" in metadata:
        logger.info("Loaded pre-trained codebook")
    else:
        logger.info(f"Codebook trained in {metadata['train_time']:.1f}s")
        logger.debug(f"Vectors for codebook clustered in {metadata['cluster_time']:.1f}s " \
                     f"and indexed in {metadata['index_time']:.1f}s")
    return asmk


def build_ivf(asmk, dataset, desc, globals, logger):
    """The second step of asmk method - building the ivf"""
    ivf_path = f"{globals['exp_path']}/ivf_{dataset}.pkl"

    asmk = asmk.build_ivf(desc['vecs'], desc['imids'], cache_path=ivf_path)

    metadata = asmk.metadata['build_ivf']
    if "load_time" in metadata:
        logger.info("Loaded indexed ivf")
    else:
        logger.info(f"Indexed descriptors in {metadata['index_time']:.2f}s")

    logger.debug(f"IVF stats: {metadata['ivf_stats']}")
    return asmk


def query_ivf(asmk, dataset, desc, globals, logger):
    """The last step of asmk method - querying the ivf"""
    metadata, _images, ranks, _scores = asmk.query_ivf(desc['qvecs'], desc['qimids'])

    logger.debug(f"Average query time (quant+aggr+search) is {metadata['query_avg_time']:.3f}s")
    gnd = f"{globals['root_path']}/features/nclt_test_nclt.pickle"
    global_metrics = evaluate(ranks, gnd, desc)
    print_results(global_metrics)
    # io_helpers.capture_stdout(lambda: compute_map_and_print(dataset, ranks.T, gnd), logger)


def demo_how(params, globals, logger):
    """Demo where asmk is applied to the HOW descriptors from eccv'20 paper, replicating reported
        results. Params is a dictionary with parameters for each step."""
    demo_params = params.pop("demo_how")
    asmk = initialize(params, demo_params, globals, logger)

    asmk = train_codebook(asmk, demo_params, globals, logger)

    # Create db and evaluate datasets
    for dataset in demo_params['eval_datasets']:
        print(f"{globals['root_path']}/features/{dataset}_" \
                                      f"{demo_params['eval_features']}.pickle")
        desc = io_helpers.load_pickle(f"{globals['root_path']}/features/{dataset}_" \
                                      f"{demo_params['eval_features']}.pickle")

        print(desc['vecs'].shape)
        logger.info(f"Loaded DB and query descriptors for {dataset}")

        asmk_dataset = build_ivf(asmk, dataset, desc, globals, logger)

        query_ivf(asmk_dataset, dataset, desc, globals, logger)


def evaluate(ranks, gnd, desc):

    global_metrics = {'tp': {r: [0] * 20 for r in [2,5,10,20]}}
    query_num = ranks.shape[0]
    map_num = ranks.shape[1]

    # query_position = np.zeros((query_num,2))
    map_positions = np.zeros((map_num,2))

    # for ndx in tqdm(range(query_num)):
    #     query_position[ndx] = np.array((desc['qcoordx'], desc['qcoordy']))

    for ndx in tqdm(range(map_num)):
        map_positions[ndx] = np.array((desc['posex'][ndx], desc['posey'][ndx]))


    for ndx in tqdm(range(query_num)):
        query_position = np.array((desc['qposex'][ndx], desc['qposey'][ndx]))
        map_top20 = ranks[ndx][:20]
        map_pos = map_positions[map_top20,...]
        delta = query_position - map_pos
        euclid_dist = np.linalg.norm(delta, axis=1) 

        global_metrics['tp'] = {r: [global_metrics['tp'][r][nn] + (1 if (euclid_dist[:nn + 1] <= r).any() else 0) for nn in range(20)] for r in [2,5,10,20]}

    global_metrics["recall"] = {r: [global_metrics['tp'][r][nn] / query_num for nn in range(20)] for r in [2,5,10,20]}

    return global_metrics


def print_results(global_metrics):
    # Global descriptor results are saved
    recall = global_metrics['recall']
    for r in recall:
        print(f"Radius: {r} [m] : ", end='')
        for x in recall[r]:
            print("{:0.3f}, ".format(x), end='')
        print("")


def main(args):
    """Argument parsing and parameter preparation for the demo"""
    # Arguments
    parser = argparse.ArgumentParser(description="ASMK demo replicating results for HOW " \
                                                 "descriptors from ECCV 2020.")
    parser.add_argument('parameters', nargs='+', type=str,
                        help="Relative path to a yaml file that contains parameters.")
    args = parser.parse_args(args)

    package_root = Path(__file__).resolve().parent.parent
    for parameters_path in args.parameters:
        # Load yaml params
        if not parameters_path.endswith(".yml"):
            parameters_path = package_root / "examples" / ("params/%s.yml" % parameters_path)
        params = io_helpers.load_params(parameters_path)

        # Resolve data folders
        globals = {}
        globals["root_path"] = (package_root / params['demo_how']['data_folder'])
        globals["root_path"].mkdir(parents=True, exist_ok=True)
        exp_name = Path(parameters_path).name[:-len(".yml")]
        globals["exp_path"] = (package_root / params['demo_how']['exp_folder']) / exp_name
        globals["exp_path"].mkdir(parents=True, exist_ok=True)

        # Setup logging
        logger = io_helpers.init_logger(globals["exp_path"] / "output.log")

        # Run demo
        demo_how(params, globals, logger)


if __name__ == "__main__":
    main(sys.argv[1:])
