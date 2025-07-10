import SimpleITK as sitk
import os
import openpyxl
import csv


# 不仅对图像进行配准，还会将变换参数作用到浮动的图像的Label上
def Rigid_registration(fixed_image, moving_image, moving_label):


    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=400)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.1)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-5, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()


    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)


    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))


    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear,
                                     float(0),
                                     moving_image.GetPixelID())

    # 对浮动图像的Label进行重采样
    moving_label_resampled = sitk.Resample(moving_label, fixed_image, final_transform, sitk.sitkNearestNeighbor,
                                     float(0),
                                     moving_label.GetPixelID())

    return moving_resampled, moving_label_resampled

# 进行自身变换
def Identity_Transform_label(fixed_image, moving_image, moving_label):


    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    # Identity Transform
    translation = sitk.TranslationTransform(3)
    translation.SetParameters((0, 0, 0, 0, 0, 0))


    moving_resampled = sitk.Resample(moving_image, fixed_image, initial_transform, sitk.sitkLinear, float(0),
                                   moving_image.GetPixelID())

    moving_label_resampled = sitk.Resample(moving_label, fixed_image, initial_transform, sitk.sitkNearestNeighbor,
                                         float(0),
                                         moving_label.GetPixelID())

    return moving_resampled, moving_label_resampled


def find_largest_file(dir_path):
    # 存储最大文件的信息
    largest_file = {'name': '', 'size': 0}
    # 遍历目录下所有文件和子文件夹
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # 构造文件的完整路径
            file_path = os.path.join(root, file)
            # 获取文件的大小
            file_size = os.path.getsize(file_path)
            # 如果文件大小更大，更新最大文件信息
            if file_size > largest_file['size']:
                largest_file['name'] = file_path
                largest_file['size'] = file_size
    # 返回最大文件的信息
    return largest_file

def find_least_file(dir_path):
    # 存储最小大文件的信息
    largest_file = {'name': '', 'size': 9999999999}
    # 遍历目录下所有文件和子文件夹
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # 构造文件的完整路径
            file_path = os.path.join(root, file)
            # 获取文件的大小
            file_size = os.path.getsize(file_path)
            # 如果文件大小更小，更新最小文件信息
            if file_size < largest_file['size']:
                largest_file['name'] = file_path
                largest_file['size'] = file_size
    # 返回最小文件的信息
    return largest_file

def find_file_in_directory(folder_path, target_filename):
    for root, dirs, files in os.walk(folder_path):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None

def find_t2_file(dir_path):
    # 存储最小大文件的信息
    dwi_file = {'name': '', 'size': 9999999999}
    # 遍历目录下所有文件和子文件夹
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # 构造文件的完整路径
            file_path = os.path.join(root, file)
            # 获取文件的大小,转换为kb
            file_size = os.path.getsize(file_path)/1024
            if file_size < 10240 and file_size > 2500:
                dwi_file['name'] = file_path
                dwi_file['size'] = file_size
    # 返回最小文件的信息
    return dwi_file

def expand_square_bounding_box(segmentation_image, target_size=64):

    # 使用LabelShapeStatisticsImageFilter获取连通区域的统计信息
    label_statistics = sitk.LabelShapeStatisticsImageFilter()
    label_statistics.Execute(segmentation_image)

    # 获取前景（标签为1）的包围盒
    bounding_box = label_statistics.GetBoundingBox(1)

    # 获取最小包围盒的尺寸
    size = bounding_box[3:5]
    max_size = max(size[:2])  # 只保证XY方向是正方形，取其中较大的尺寸

    # 计算外扩后的边长
    target_size = max(target_size, max_size)

    # 计算额外的边长
    extra_size = (target_size - max_size) // 2

    # 计算新的起始点
    start_point = [bounding_box[0] - extra_size, bounding_box[1] - extra_size, bounding_box[2]]

    # 构建新的包围盒信息 [xmin, ymin, zmin, xsize, ysize, zsize]
    expanded_box = start_point + [target_size, target_size, bounding_box[5]]

    return expanded_box

def crop_to_bounding_box(image, bounding_box):
    # 获取最小包围盒的起始点和尺寸
    start_point = bounding_box[:3]
    size = bounding_box[3:6]

    # 获取图像的大小
    image_size = image.GetSize()



    # 调整裁剪区域，确保不超出图像范围
    for i in range(3):
        if start_point[i] < 0:
            return None;
            # size[i] += start_point[i]
            # start_point[i] = 0
        if start_point[i] + size[i] > image_size[i]:
            return None
            # size[i] = image_size[i] - start_point[i]

    # 使用SimpleITK的裁剪功能对图像进行裁剪
    cropped_image = sitk.RegionOfInterest(image, size, start_point)

    return cropped_image

def get_label_of_image(excel_file, studyUid):
    # 打开Excel文件
    workbook = openpyxl.load_workbook(excel_file)

    # 获取第一个工作表（Sheet）
    sheet = workbook.active

    # 遍历每一行（跳过标题行）
    for row in sheet.iter_rows(min_row=2, values_only=True):

        # row[3]是studyUid，判断是否相等
        if str(row[3]) == studyUid:
            # row[4]是分类标签
            output = row[4]
            # 关闭Excel文件
            workbook.close()
            return output
    # 关闭Excel文件
    workbook.close()
    return None

def resample_image(image, new_spacing):
    """
    对输入图像进行重采样。

    参数：
        image (SimpleITK.Image): 需要被重采样的输入图像。
        new_spacing (tuple): 新的像素间距 (深度，高度，宽度)。

    返回：
        SimpleITK.Image: 重采样后的图像。
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)

    resampled_image = resampler.Execute(image)
    return resampled_image

# 对二值图像进行重采样，采用的是sitkNearestNeighbor插值方式
def resample_binary_image(image, new_spacing):

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)

    resampled_image = resampler.Execute(image)
    return resampled_image

# 读取数据的路径
origin_dir = r'F:\BreastMR-Data\origin-data\PKUPH\PKUPH-1291(b0)'

# 读取label文件
excel_file = r'F:\BreastMR-Data\new-preprocess\PKUPH\clinical_data.xlsx'

# 指定保存路径的文件夹
b0_output_folder = r'F:\BreastMR-Data\new-preprocess\PKUPH\b0'
b800_output_folder = r'F:\BreastMR-Data\new-preprocess\PKUPH\b800'
t2_output_folder = r'F:\BreastMR-Data\new-preprocess\PKUPH\t2'
c_output_folder = r'F:\BreastMR-Data\new-preprocess\PKUPH\c'
segment_ouput_folder = r'F:\BreastMR-Data\new-preprocess\PKUPH\segment'

# 保存标签的文件
csv_file = r'F:\BreastMR-Data\new-preprocess\PKUPH\label.csv'



if not os.path.exists(b0_output_folder):
    os.makedirs(b0_output_folder)
    os.makedirs(b800_output_folder)
    os.makedirs(t2_output_folder)
    os.makedirs(c_output_folder)
    os.makedirs(segment_ouput_folder)

dataList = []
labelList = []


def main():
    for data in os.listdir(origin_dir):

        # print(data)

        # 获取原始图像路径
        origin_image_dir = os.path.join(origin_dir, data)

        # 找到最大文件，最大的图像一定是C增强的图像
        c_image_dir = find_largest_file(origin_image_dir)['name']
        # 获取C图像的路径
        CImageDir = os.path.split(c_image_dir)[0]
        # 该路径下最小的文件为C的Mask
        c_segment_dir = find_least_file(CImageDir)['name']


        # 找到名字包含b0/b800的数据
        b0_image_dir = find_file_in_directory(origin_image_dir, "b0.nii.gz")
        b800_image_dir = find_file_in_directory(origin_image_dir, "b800.nii.gz")
        # 获取DWI图像的路径
        DWIImageDir = os.path.split(b800_image_dir)[0]
        # 该路径下最小的文件为DWI的Mask
        dwi_segment_dir = find_least_file(DWIImageDir)['name']


        # 找到T2的图像，根据图像尺寸
        t2_image_dir = find_t2_file(origin_image_dir)['name']
        # 获取T2图像的路径
        T2ImageDir = os.path.split(t2_image_dir)[0]
        # 该路径下最小的文件为t2的Mask
        t2_segment_dir = find_least_file(T2ImageDir)['name']


        # 读取分割图像及各个序列图像
        c_segment_image = sitk.ReadImage(c_segment_dir)
        dwi_segment_image = sitk.ReadImage(dwi_segment_dir)
        t2_segment_image = sitk.ReadImage(t2_segment_dir)


        b0_image = sitk.ReadImage(b0_image_dir, sitk.sitkFloat32)
        b800_image = sitk.ReadImage(b800_image_dir, sitk.sitkFloat32)
        t2_image = sitk.ReadImage(t2_image_dir, sitk.sitkFloat32)
        c_image = sitk.ReadImage(c_image_dir, sitk.sitkFloat32)



        # 重采样成1*1*1
        new_spacing = (1.0, 1.0, 1.0)
        resample_c_segment_image = resample_binary_image(c_segment_image, new_spacing)
        resample_dwi_segment_image = resample_binary_image(dwi_segment_image, new_spacing)
        resample_t2_segment_image = resample_binary_image(t2_segment_image, new_spacing)

        resample_c_image = resample_image(c_image, new_spacing)
        resample_b0_image = resample_image(b0_image, new_spacing)
        resample_b800_image = resample_image(b800_image, new_spacing)
        resample_t2_image = resample_image(t2_image, new_spacing)



        # 将dwi图像作为参考图像，将t2/c原始图像及label配准到dwi序列上
        # c_regis_image, c_regis_segment_image = Rigid_registration(resample_b800_image, resample_c_image, resample_c_segment_image)
        # t2_regis_image, t2_regis_segment_image = Rigid_registration(resample_b800_image, resample_t2_image, resample_t2_segment_image)
        c_regis_image = resample_c_image
        c_regis_segment_image = resample_c_segment_image
        t2_regis_image = resample_t2_image
        t2_regis_segment_image = resample_t2_segment_image


        # 获取最小方形包围盒（X、Y方形）
        c_bounding_box = expand_square_bounding_box(c_regis_segment_image)
        dwi_bounding_box = expand_square_bounding_box(resample_dwi_segment_image)
        t2_bounding_box = expand_square_bounding_box(t2_regis_segment_image)

        # 裁剪分割图像
        cropped_c_segmentation = crop_to_bounding_box(c_regis_segment_image, c_bounding_box)
        cropped_dwi_segmentation = crop_to_bounding_box(resample_dwi_segment_image, dwi_bounding_box)
        cropped_t2_segmentation = crop_to_bounding_box(t2_regis_segment_image, t2_bounding_box)

        if cropped_c_segmentation == None or cropped_t2_segmentation == None or cropped_dwi_segmentation == None:
            print(data)
            continue

        # 裁剪原始图像
        cropped_c = crop_to_bounding_box(c_regis_image, c_bounding_box)
        cropped_b0 = crop_to_bounding_box(resample_b0_image, dwi_bounding_box)
        cropped_b800 = crop_to_bounding_box(resample_b800_image, dwi_bounding_box)
        cropped_t2 = crop_to_bounding_box(t2_regis_image, t2_bounding_box)



        # 生成保存路径和文件名
        save_path_segmentation = os.path.join(segment_ouput_folder, f"{data}.nii.gz")

        save_path_c = os.path.join(c_output_folder, f"{data}_c.nii.gz")
        save_path_b0 = os.path.join(b0_output_folder, f"{data}_b0.nii.gz")
        save_path_b800 = os.path.join(b800_output_folder, f"{data}_b800.nii.gz")
        save_path_t2 = os.path.join(t2_output_folder, f"{data}_t2.nii.gz")


        # 保存裁剪后的分割图像
        sitk.WriteImage(cropped_dwi_segmentation, save_path_segmentation)

        # 保存裁剪后的原始图像
        sitk.WriteImage(cropped_c, save_path_c)
        sitk.WriteImage(cropped_b0, save_path_b0)
        sitk.WriteImage(cropped_b800, save_path_b800)
        sitk.WriteImage(cropped_t2, save_path_t2)


        # 根据studyUid 来获取图像的分类标签
        classfication_label = get_label_of_image(excel_file, data)
        if classfication_label != None:
            dataList.append(data)
            labelList.append(classfication_label)

    # 将数据保存为CSV文件
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)

        # 写入标题行
        writer.writerow(["studyUid", "Label"])

        # 将两个列表的数据同时写入CSV文件的两列
        writer.writerows(zip(dataList, labelList))



if __name__ == "__main__":
    main()