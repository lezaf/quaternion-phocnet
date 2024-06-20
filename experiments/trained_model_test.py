from   src.models.quaternion_phocnet_2  import QuaternionPHOCNet
from   src.utils.save_load              import my_torch_load
from   skimage                          import io                 as img_io
import numpy                            as     np
from   matplotlib                       import pyplot             as plt
import torch


def calculate_output_neurons(phoc_unigram_levels_list):
    # TODO: maybe change hardcoded number of characters, if needed
    # 36: a -> z and 0 -> 9
    return sum(36*level for level in phoc_unigram_levels_list)

n_out = calculate_output_neurons([1,2,4])
n_out = n_out + 4 - n_out%4 if n_out%4 != 0 else n_out

qcnn = QuaternionPHOCNet(n_out=n_out,
                         in_channels=4,
                         gpp_type='spp',
                         pooling_levels=3)

my_torch_load(qcnn, './saved_models/Quaternion-PHOCNet_2.pt')

qcnn.eval()

# Load a query image and output corresponding phoc

word_img_name = '270-08-04.png'
word_img = img_io.imread(word_img_name)

# Scale black pixels to 1 and white to 0
word_img = 1 - word_img.astype(np.float32)
word_img = word_img.reshape((1,1,) + word_img.shape) if len(word_img.shape) == 2 else word_img.reshape((1,) + word_img.shape)
word_img = torch.from_numpy(word_img)

# If grayscale with 1 channel, make it 3 channels
word_img = word_img.expand(-1,3,-1,-1) if word_img.size(1) == 1 else word_img

# Make the image to be of quaternion format
new_word_img = torch.zeros(word_img.size(0), 4, word_img.size(2), word_img.size(3))
new_word_img[:, 1:, :, :] = word_img[:,:,:,:]

word_img = new_word_img

# Pass the image from the qcnn
output = torch.sigmoid(qcnn(word_img))

torch.set_printoptions(precision=2)
print(output)

#img_io.imshow(word_img[0,:,:,:].permute(1, 2, 0).numpy())
#plt.show()