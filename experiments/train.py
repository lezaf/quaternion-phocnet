import  argparse
import  logging
import  copy
import  tqdm
import  numpy                              as     np

import  torch.nn                           as     nn
import  torch.cuda
import  torch.optim
import  torch.autograd
from    torch.utils.data                   import DataLoader
from    torch.utils.data.sampler           import WeightedRandomSampler

from    experiments.dataset_loader.gw_alt  import GWDataset
from    experiments.dataset_loader.iam_alt import IAMDataset
from    src.models.quaternion_phocnet_2    import QuaternionPHOCNet
from    src.utils.save_load                import my_torch_load, my_torch_save
from    src.losses.cosine_loss             import CosineLoss
from    src.evaluation.retrieval           import map_from_feature_matrix, map_from_query_test_feature_matrices

'''
TODO-list:
    * transform following hardcoded variables to cmd arguments:
        - augmentation
        - n_train_images
        - load_pretrained
        - pretrained_name
'''

# Arguments helper functions
def learning_rate_step_parser(lrs_string):
    return [(int(elem.split(':')[0]), float(elem.split(':')[1])) for elem in lrs_string.split(',')]


def save_name_check(save_name):
    if save_name.split('.')[-1] != 'pt':
        raise TypeError


# Training happens here
def train():
    augmentation = False
    n_train_images = 500000
    #n_train_images = 30000

    load_pretrained = False
    pretrained_name = ''
    loss_selection = 'BCE'
    # save_model_name = 'Quaternion-PHOCNet_6.pt' # Transformed in cmd argument

    logger = logging.getLogger('Quaternion-PHOCNet-Experiment::train')
    logger.info('--- Running Quaternion PHOCNet Training ---')
    args_parser = argparse.ArgumentParser()

    # TRAINING ARGUMENTS
    args_parser.add_argument('--learning_rate_step', '-lrs', type=learning_rate_step_parser, default='30000:1e-4,50000:1e-5',
                        help='A dictionary-like string indicating the learning rate for up to the number of iterations. ' +
                             'E.g. the default \'60000:1e-4,100000:1e-5\' means learning rate 1e-4 up to step 60000 and 1e-5 till 100000.')
    args_parser.add_argument('--momentum', '-mom', action='store', type=float, default=0.9,
                        help='The momentum for SGD training (or beta1 for Adam). Default: 0.9')
    #parser.add_argument('--momentum2', '-mom2', action='store', type=float, default=0.999,
    #                    help='Beta2 if solver is Adam. Default: 0.999')
    #parser.add_argument('--delta', action='store', type=float, default=1e-8,
    #                    help='Epsilon if solver is Adam. Default: 1e-8')
    args_parser.add_argument('--solver_type', '-st', choices=['SGD', 'Adam'], default='Adam',
                        help='Which solver type to use. Possible: SGD, Adam. Default: Adam')
    args_parser.add_argument('--display', action='store', type=int, default=500,
                        help='The number of iterations after which to display the loss values. Default: 500')
    args_parser.add_argument('--test_interval', action='store', type=int, default=2000,
                        help='The number of iterations after which to periodically evaluate the PHOCNet. Default: 2000')
    args_parser.add_argument('--iter_size', '-is', action='store', type=int, default=10,
                        help='The batch size after which the gradient is computed. Default: 10')
    args_parser.add_argument('--batch_size', '-bs', action='store', type=int, default=1,
                        help='The batch size after which the gradient is computed. Default: 1')
    args_parser.add_argument('--weight_decay', '-wd', action='store', type=float, default=0.00005,
                        help='The weight decay for SGD training. Default: 0.00005')
    args_parser.add_argument('--gpu_id', '-gpu', action='store',
                        type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                        default='0',
                        help='The ID of the GPU to use. If not specified, training is run in CPU mode.')

    # EXPERIMENT ARGUMENTS
    args_parser.add_argument('--min_image_width_height', '-miwh', action='store', type=int, default=26,
                        help='The minimum width or height of the images that are being fed to the AttributeCNN. Default: 26')
    args_parser.add_argument('--phoc_unigram_levels', '-pul',
                        action='store',
                        type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                        default='1,2,4,8',
                        help='The comma seperated list of PHOC unigram levels. Default: 1,2,4,8')
    args_parser.add_argument('--embedding_type', '-et', action='store',
                        choices=['phoc', 'spoc', 'dctow', 'phoc-ppmi', 'phoc-pruned'],
                        default='phoc',
                        help='The label embedding type to be used. Possible: phoc, spoc, phoc-ppmi, phoc-pruned. Default: phoc')
    args_parser.add_argument('--fixed_image_size', '-fim', action='store',
                        type=lambda str_tuple: tuple([int(elem) for elem in str_tuple.split(',')]),
                        default=None ,
                        help='Specifies the images to be resized to a fixed size when presented to the CNN. Argument must be two comma seperated numbers.')
    args_parser.add_argument('--dataset', '-ds', required=True, choices=['gw','iam'],
                        help='The dataset to be trained on.')
    args_parser.add_argument('--save_name', '-sn', required=True, action='store', type=save_name_check,
                        help='The name the trained model will be saved with. Extension: .pt')

    args = args_parser.parse_args()

    # Check CUDA environment availability
    if not torch.cuda.is_available():
        logger.warning('Could not find CUDA environment, using CPU mode')
        args.gpu_id = None

    # Print arguments
    logger.info('###########################################')
    logger.info('Experiment Parameters:')
    for key, value in vars(args).items():
        logger.info('%s: %s', str(key), str(value))
    logger.info('###########################################')

    # Print if augmentation is True/False
    logger.info('Augmentation: %s', str(augmentation))

    logger.info('Loading dataset %s...', args.dataset)
    if args.dataset == 'gw':
        train_set = GWDataset(gw_root_dir='./data/gw4860',      # NOTE: Changed from '..' to '.' for Google Colab
                              cv_split_method='almazan',
                              cv_split_idx=1,
                              image_extension='.tif',
                              embedding=args.embedding_type,
                              phoc_unigram_levels=args.phoc_unigram_levels,
                              fixed_image_size=args.fixed_image_size,
                              min_image_width_height=args.min_image_width_height)

    if args.dataset == 'iam':
        train_set = IAMDataset(gw_root_dir='../data/IAM',
                               image_extension='.png',
                               embedding=args.embedding_type,
                               phoc_unigram_levels=args.phoc_unigram_levels,
                               fixed_image_size=args.fixed_image_size,
                               min_image_width_height=args.min_image_width_height)

    test_set = copy.copy(train_set)

    # Create train/test partitions according to '(...).cv.indices.npy' file specified
    train_set.mainLoader(partition='train')
    test_set.mainLoader(partition='test', transforms=None)

    # Create the train DataLoader object over the dataset
    if augmentation: # TODO: argparse
        train_loader = DataLoader(train_set,
                                  sampler=WeightedRandomSampler(train_set.weights, n_train_images), # TODO: argparse
                                  batch_size=args.batch_size,
                                  num_workers=8)
    else:
        train_loader = DataLoader(train_set,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=8)

    test_loader = DataLoader(test_set,
                             batch_size=1,
                             shuffle=False,
                             num_workers=8)

    train_loader_iter = iter(train_loader)

    # Create the Quaternion PHOCNet
    logger.info('Preparing Quaternion-PHOCNet...')

    ''' Added by @lezaf.
    Output of last fully connected layer should be multiple of 4.
    If phoc descriptor isn't multiple of 4, add the modulo as padding.
    '''
    n_out = train_set[0][1].shape[0]
    n_out = n_out + 4 - n_out%4 if n_out%4 != 0 else n_out

    qcnn = QuaternionPHOCNet(n_out=n_out,
                             in_channels=4,
                             gpp_type='spp',
                             #pooling_levels=([1], [5])
                             pooling_levels=3)

    # Load pretrained model
    if load_pretrained: # TODO: argparse
        my_torch_load(qcnn, pretrained_name) # TODO: argparse

    if loss_selection == 'BCE':
        loss = nn.BCEWithLogitsLoss(size_average=True)
    elif loss_selection == 'cosine':
        # TODO: why not to use CosineLoss from pytorch?
        loss = CosineLoss(size_average=False, use_sigmoid=True)
    else:
        raise ValueError('Not supported loss function.\nSupported now: BCE, cosine.\nFound: ',loss_selection)

    # Enable GPU for QCNN training
    if args.gpu_id is not None:
        if len(args.gpu_id) > 1:
            qcnn = nn.DataParallel(qcnn, device_ids=args.gpu_id)
            qcnn.cuda()
        else:
            qcnn.cuda(args.gpu_id[0])

    # Create optimizer
    if args.solver_type == 'SGD':
        optimizer = torch.optim.SGD(qcnn.parameters(),
                                    args.learning_rate_step[0][1],
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    if args.solver_type == 'Adam':
        optimizer = torch.optim.Adam(qcnn.parameters(),
                                     args.learning_rate_step[0][1],
                                     weight_decay=args.weight_decay)

    lr_cnt = 0
    max_iters = args.learning_rate_step[-1][0]

    # Start training
    optimizer.zero_grad()
    logger.info('Training:')
    for iter_idx in range(max_iters):
        # Evaluate network every 'test_interval' iterations
        if iter_idx % args.test_interval == 0 and iter_idx > 0:
            logger.info('Evaluating net after %d iterations', iter_idx)
            evaluate_cnn(qcnn=qcnn,
                         dataset_loader=test_loader,
                         args=args)

        for _ in range(args.iter_size):
            try:
                word_img, embedding, _, _ = train_loader_iter.next()
            except StopIteration:
                logger.info('Resetting data loader')
                train_loader_iter = iter(train_loader)
                word_img, embedding, _, _ = train_loader_iter.next()

            ''' Added by @lezaf. '''
            # If image is grayscale with one channel, transform it to have 3 (same) channels plus 1 zero channel
            word_img = word_img.expand(-1,4,-1,-1) if word_img.size(1) == 1 else word_img

            # Make the image to be of quaternion format with real part = 0
            if word_img.size(1) < 4:
                new_word_img = torch.zeros(word_img.size(0), 4, word_img.size(2), word_img.size(3))
                new_word_img[:, 1:, :, :] = word_img[:,:,:,:]

                word_img = new_word_img

            if args.gpu_id is not None:
                if len(args.gpu_id) > 1:
                    word_img = word_img.cuda()
                    embedding = embedding.cuda()
                else:
                    word_img = word_img.cuda(args.gpu_id[0])
                    embedding = embedding.cuda(args.gpu_id[0])

            word_img = torch.autograd.Variable(word_img)
            embedding = torch.autograd.Variable(embedding)
            output = qcnn(word_img)
            ''' BCEloss ??? '''
            loss_val = loss(output, embedding)*args.batch_size
            loss_val.backward()
        optimizer.step()
        optimizer.zero_grad()

        # mean running errors??
        if (iter_idx+1) % args.display == 0:
            logger.info('Iteration %*d: %f', len(str(max_iters)), iter_idx+1, loss_val.data)    # CHANGED_ORIGINAL: loss_val.data[0] -> loss_val.data

        # change lr
        if (iter_idx + 1) == args.learning_rate_step[lr_cnt][0] and (iter_idx+1) != max_iters:
            lr_cnt += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate_step[lr_cnt][1]

        #if (iter_idx + 1) % 10000 == 0:
        #    torch.save(cnn.state_dict(), 'PHOCNet.pt')
            # .. to load your previously training model:
            #cnn.load_state_dict(torch.load('PHOCNet.pt'))

    #torch.save(cnn.state_dict(), 'PHOCNet.pt')
    my_torch_save(qcnn, args.save_name)


def evaluate_cnn(qcnn, dataset_loader, args):
    logger = logging.getLogger('Quaternion-PHOCNet-Experiment::test')

    # Set CNN in eval mode
    qcnn.eval()

    logger.info('Computing net output:')
    qry_ids = [] #np.zeros(len(dataset_loader), dtype=np.int32)
    class_ids = np.zeros(len(dataset_loader), dtype=np.int32)
    embedding_size = dataset_loader.dataset.embedding_size()
    embeddings = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)
    outputs = np.zeros((len(dataset_loader), embedding_size), dtype=np.float32)

    for sample_idx, (word_img, embedding, class_id, is_query) in enumerate(tqdm.tqdm(dataset_loader)):
        ''' Added by @lezaf. '''
        # If image is grayscale with one channel, transform it to have 3 (same) channels
        # and make it type double() in any case
        word_img = word_img.expand(-1,3,-1,-1) if word_img.size(1) == 1 else word_img

        if word_img.size(1) < 4:
            new_word_img = torch.zeros(word_img.size(0), 4, word_img.size(2), word_img.size(3))
            new_word_img[:, 1:, :, :] = word_img[:,:,:,:]

            word_img = new_word_img

        if args.gpu_id is not None:
            # in one gpu!!
            word_img = word_img.cuda(args.gpu_id[0])
            embedding = embedding.cuda(args.gpu_id[0])
            #word_img, embedding = word_img.cuda(args.gpu_id), embedding.cuda(args.gpu_id)

        word_img = torch.autograd.Variable(word_img)
        embedding = torch.autograd.Variable(embedding)
        ''' BCEloss ??? '''

        output = qcnn(word_img)

        outputs[sample_idx] = output.data.cpu().numpy().flatten()
        embeddings[sample_idx] = embedding.data.cpu().numpy().flatten()
        class_ids[sample_idx] = class_id.numpy()[0,0]

        if is_query[0] == 1:
            qry_ids.append(sample_idx)  #[sample_idx] = is_query[0]

    '''
    # find queries

    unique_class_ids, counts = np.unique(class_ids, return_counts=True)
    qry_class_ids = unique_class_ids[np.where(counts > 1)[0]]

    # remove stopwords if needed

    qry_ids = [i for i in range(len(class_ids)) if class_ids[i] in qry_class_ids]
    '''

    qry_outputs = outputs[qry_ids][:]
    qry_class_ids = class_ids[qry_ids]

    # run word spotting
    logger.info('Computing mAPs...')

    mAP, ave_precs_qbe = map_from_query_test_feature_matrices(query_features=qry_outputs,
                                                              test_features=outputs,
                                                              query_labels=qry_class_ids,
                                                              test_labels=class_ids,
                                                              metric='cosine',
                                                              drop_first=True)

    logger.info('mAP: %3.2f', np.mean(ave_precs_qbe[ave_precs_qbe > 0])*100)
    #logger.info('mAP: %3.2f', mAP)

    # Clean up and set CNN in train mode again
    qcnn.train()


if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s, %(levelname)s, %(name)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    train()