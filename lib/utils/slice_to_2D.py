import os 
import SimpleITK as sitk
import matplotlib.pyplot as plt
raw_dir= './raw_data'
b800_dir = raw_dir + '/' + 'b800'
t1c_dir = raw_dir + '/' + 't1c'
segment_dir = raw_dir + '/' + 'segment'

b800_output_dir = b800_dir.replace('raw_data', 'data')
t1c_output_dir = t1c_dir.replace('raw_data', 'data')
segment_output_dir = segment_dir.replace('raw_data', 'data')

if not os.path.exists(segment_output_dir):
    os.makedirs(segment_output_dir)
for data in os.listdir(segment_dir):
    sub_dir = data.split('.')[0]
    dir = os.path.join(segment_output_dir, sub_dir)
    if not os.path.exists(dir):
        os.mkdir(dir)
    print(sub_dir)
    file_path = os.path.join(segment_dir, data)
    image = sitk.ReadImage(file_path)
    numpy_image = sitk.GetArrayFromImage(image)
    for i in range(numpy_image.shape[0]):
        fig = numpy_image[i,:,:]
        plt.imsave(os.path.join(dir, '{}.png'.format(i)),fig, cmap='gray', vmin=0, vmax=1)

