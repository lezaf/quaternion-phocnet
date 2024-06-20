import  logging
import  argparse
import  tqdm
import  torch
import  numpy                              as     np

from    torch.utils.data                   import DataLoader

from    experiments.dataset_loader.gw_alt  import GWDataset
from    experiments.dataset_loader.iam_alt import IAMDataset
from    src.models.quaternion_phocnet_2    import QuaternionPHOCNet
from    src.models.myphocnet               import PHOCNet
from    src.utils.save_load                import my_torch_load
from    src.evaluation.retrieval           import map_from_query_test_feature_matrices


def evaluate_cnn(cnn, test_loader, model_type, gpu_id, metric):
    logger = logging.getLogger('Quaternion-PHOCNet-Experiment::test')

    # Set CNN in eval mode
    cnn.eval()

    logger.info('Computing net output:')
    qry_ids = [] #np.zeros(len(test_loader), dtype=np.int32)
    class_ids = np.zeros(len(test_loader), dtype=np.int32)
    embedding_size = test_loader.dataset.embedding_size()
    embeddings = np.zeros((len(test_loader), embedding_size), dtype=np.float32)
    outputs = np.zeros((len(test_loader), embedding_size), dtype=np.float32)

    for sample_idx, (word_img, embedding, class_id, is_query) in enumerate(tqdm.tqdm(test_loader)):
        ''' Added by @lezaf. '''
        # If model type is 'Quaternion PHOCNet', make image quaternionic
        if model_type == 'q':
            word_img = word_img.expand(-1,3,-1,-1) if word_img.size(1) == 1 else word_img

            if word_img.size(1) < 4:
                new_word_img = torch.zeros(word_img.size(0), 4, word_img.size(2), word_img.size(3))
                new_word_img[:, 1:, :, :] = word_img[:,:,:,:]

                word_img = new_word_img

        if gpu_id is not None:
            # in one gpu!!
            word_img = word_img.cuda(gpu_id)
            embedding = embedding.cuda(gpu_id)

        word_img = torch.autograd.Variable(word_img)
        embedding = torch.autograd.Variable(embedding)
        ''' BCEloss ??? '''

        output = cnn(word_img)

        outputs[sample_idx] = output.data.cpu().numpy().flatten()
        embeddings[sample_idx] = embedding.data.cpu().numpy().flatten()
        class_ids[sample_idx] = class_id.numpy()[0,0]

        if is_query[0] == 1:
            qry_ids.append(sample_idx)  #[sample_idx] = is_query[0]

    qry_outputs = outputs[qry_ids][:]
    qry_class_ids = class_ids[qry_ids]

    # run word spotting
    logger.info('Computing mAPs...')

    # TODO: don't know if 'mAP' is useful
    mAP, ave_precs_qbe = map_from_query_test_feature_matrices(query_features=qry_outputs,
                                                              test_features=outputs,
                                                              query_labels=qry_class_ids,
                                                              test_labels=class_ids,
                                                              metric=metric,
                                                              drop_first=True)

    logger.info('mAP: %3.2f', np.mean(ave_precs_qbe[ave_precs_qbe > 0])*100)


def main():
    n_out_temp = 252

    logger = logging.getLogger('KWS::evaluate-cnn')
    logger.info('--- Running evaluation process ---')

    # Define arguments
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--dataset', '-ds', required=True, choices=['gw', 'iam'],
                             help='The dataset to evaluate on. Options: gw, iam')
    args_parser.add_argument('--phoc_unigram_levels', '-pul', action='store',
                             type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                             default='1,2,4,8',
                             help='The comma seperated list of PHOC unigram levels. Default: 1,2,4,8')
    args_parser.add_argument('--model_type', '-mt', required=True, choices=['p', 'q'],
                             help='Type of model to use. Options: p (original PHOCNet, q (Quaternion PHOCNet)).')
    args_parser.add_argument('--phocnet_model_path', '-m', required=True,
                             help='The path of model to load.')
    args_parser.add_argument('--metric', '-metric', required=True, choices=['bc', 'co'],
                             help='Metric to use for evaluation. Options: bc (bray-curtis), co (cosine).')
    args_parser.add_argument('--gpu_id', '-gpu', default=None, type=int,
                             help='The ID of the GPU to use. If not specified, running on CPU.')

    args = args_parser.parse_args()

    # Check CUDA environment availability
    if not torch.cuda.is_available():
        logger.warning('Could not find CUDA environment, using CPU mode')
        args.gpu_id = None

    '''
        TODO - parameterize following hardcoded values
            * embedding
            * fixed_image_size
            * min_image_width_height
    '''
    logger.info('Loading dataset %s...', args.dataset)
    if args.dataset == 'gw':
        test_set = GWDataset(gw_root_dir='../data/gw4860',
                             cv_split_method='almazan',
                             cv_split_idx=1,
                             image_extension='.tif',
                             embedding='phoc',
                             phoc_unigram_levels=args.phoc_unigram_levels,
                             fixed_image_size=None,
                             min_image_width_height=26)

    if args.dataset == 'iam':
        test_set = IAMDataset(gw_root_dir='./data/IAM',
                              image_extension='.png',
                              embedding='phoc',
                              phoc_unigram_levels=args.phoc_unigram_levels,
                              fixed_image_size=None,
                              min_image_width_height=26)

    # Load test set
    test_set.mainLoader(partition='test', transforms=None)

    # Define DataLoader for test set
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)

    # Load trained CNN
    # Load original PHOCNet
    if args.model_type == 'p':
        logger.info('Loading original trained PHOCNet..')
        cnn = PHOCNet(n_out=n_out_temp,
                      input_channels=1,
                      gpp_type='spp',
                      pooling_levels=3)
    # Load Quaternion PHOCNet
    elif args.model_type == 'q':
        logger.info('Loading trained Quaternion-PHOCNet..')
        cnn = QuaternionPHOCNet(n_out=n_out_temp,
                                in_channels=4,
                                gpp_type='spp',
                                pooling_levels=3)

    my_torch_load(cnn, args.phocnet_model_path)

    # Enable cuda for cnn model
    if args.gpu_id is not None:
        cnn.cuda(args.gpu_id)

    # Decide metric
    if args.metric == 'bc':
        metric = 'braycurtis'
    elif args.metric == 'co':
        metric = 'cosine'

    # Evaluate CNN
    evaluate_cnn(cnn, test_loader, args.model_type, args.gpu_id, metric)


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    main()
