import os
import random
import re
import json
from PIL import ImageFile
from PIL import Image
from PIL import ImageDraw
import pickle
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

""" VideoFrameDataset """

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_loader(path):
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except Exception as e:
        print(f"警告: 无法加载图片 {path}, 错误: {e}")
        # 返回一张全黑的图或其他占位图，或者返回 None
        return Image.new('RGB', (256, 256), color='black')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    '''
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
    '''
    return pil_loader(path)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def height_index_from_frame_name(fname, fallback):
    stem = os.path.splitext(os.path.basename(fname))[0]
    digits = ''.join(ch for ch in stem if ch.isdigit())
    if digits == '':
        return fallback
    return int(digits)

def find_classes(dir):
    cache_path = os.path.join(dir, '.classes_cache.pkl')
    
    # 如果已经缓存过，直接读取，瞬间完成
    if os.path.exists(cache_path):
        print("Loading classes from cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
            
    print("Caching class indices, this only runs once...")
    assert(os.path.exists(dir)), f'{dir} does not exist'
    
    # 优化：直接使用 os.scandir，它比 os.listdir 快得多，且不需要额外做 isdir 判断
    classes = []
    with os.scandir(dir) as it:
        for entry in it:
            if entry.is_dir():
                classes.append(entry.name)
    
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    
    # 存入缓存
    with open(cache_path, 'wb') as f:
        pickle.dump((classes, class_to_idx), f)
        
    return classes, class_to_idx

def find_classes_rm(dir):
    # 如果你的数据不需要分类（所有视频都是同一类），可以设定为默认值
    # 这样 class_to_idx 只有一个条目
    classes = ['default']
    class_to_idx = {'default': 0}
    
    # 将结果存入缓存，保持原有逻辑一致性
    cache_path = os.path.join(dir, '.classes_cache.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump((classes, class_to_idx), f)
        
    return classes, class_to_idx

def class_name_to_idx(annotation_dir):
    """
    return class indices from 0 ~ num_classes-1
    """
    fpath = os.path.join(annotation_dir, "classInd.txt")
    with open(fpath, "r") as f:
        data = f.readlines()
        class_to_idx = {x.strip().split(" ")[1].lower():int(x.strip().split(" ")[0]) - 1 for x in data}
    return class_to_idx


def make_dataset(dir, nframes, class_to_idx, frame_stride=1, **kwargs):
    """
    videos are saved in second-level directory:
    dir: video dir. Format:
        videoxxx
            videoxxx_1
                frame1.jpg
                frame2.jpg
            videoxxx_2
                frame1.jpg
                ...
        videoxxx
        
    nframes: num of frames of every video clips
    class_to_idx: for mapping video name to video id
    """
    if frame_stride != 1:
        raise NotImplementedError
    
    clips = []
    videos = []
    n_clip = 0
    video_frames = []
    for video_name in sorted(os.listdir(dir)):
        if os.path.isdir(os.path.join(dir,video_name)):
            
            # eg: dir + '/rM7aPu9WV2Q'
            subfolder_path = os.path.join(dir, video_name) # video_name: rM7aPu9WV2Q
            for subsubfold in sorted(os.listdir(subfolder_path)):
                subsubfolder_path = os.path.join(subfolder_path, subsubfold)
                if os.path.isdir(subsubfolder_path): # eg: dir/rM7aPu9WV2Q/1'
                    clip_frames = []
                    i = 1
                    # traverse frames in one video
                    for fname in sorted(os.listdir(subsubfolder_path)):
                        if is_image_file(fname):
                            img_path = os.path.join(subsubfolder_path, fname) # eg: dir + '/rM7aPu9WV2Q/rM7aPu9WV2Q_1/rM7aPu9WV2Q_frames_00086552.jpg'
                            frame_info = (img_path, class_to_idx[video_name]) #(img_path, video_id)
                            clip_frames.append(frame_info)
                            video_frames.append(frame_info)
                            
                            # append clips, clip_step=n_frames (no frame overlap between clips).
                            if i % nframes == 0 and i >0:
                                clips.append(clip_frames)
                                n_clip += 1
                                clip_frames = []
                            i = i+1
                    
                    if len(video_frames) >= nframes:
                        videos.append(video_frames)
                    video_frames = []

    print('number of long videos:', len(videos))
    print('number of short videos', len(clips))
    return clips, videos

def split_by_captical(s):
    s_list = re.sub( r"([A-Z])", r" \1", s).split()
    string = ""
    for s in s_list:
        string += s + " "
    return string.rstrip(" ").lower()

def make_dataset_ucf(dir, nframes, class_to_idx, frame_stride=1, clip_step=None):
    if clip_step is None:
        clip_step = nframes
    
    clips = []
    videos = []
    count = 0

    # 1. 使用 scandir 获取目录，速度比 listdir 快得多
    # 2. 避免使用 sorted()，对于 11 万个项目，这会造成巨大延迟
    print("正在扫描数据集，请稍候...")
    with os.scandir(dir) as it:
        for entry in it:
            # 过滤非目录文件，并排除隐藏文件（如 .DS_Store）
            if entry.is_dir() and not entry.name.startswith('.'):
                video_path = entry.path
                
                # 预读该文件夹下所有图片（如果这里卡住，说明磁盘IO太慢）
                # 为了速度，这里改为只获取文件名列表
                try:
                    fnames = [f for f in os.listdir(video_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    fnames.sort(key=natural_key) # 对单个视频内的帧排序开销较小
                except Exception:
                    continue # 如果文件夹读取出错，直接跳过
                
                if not fnames:
                    continue

                frames = []
                for fname in fnames:
                    img_path = os.path.join(video_path, fname)
                    frame_info = {
                        "img_path": img_path, 
                        "class_index": class_to_idx.get(entry.name, 0),
                        "class_name": entry.name,
                        "class_caption": entry.name
                    }
                    frames.append(frame_info)
                
                frames = frames[::frame_stride]
                
                if len(frames) >= nframes:
                    videos.append(frames)
                    # make clips
                    start_indices = range(0, len(frames) - nframes + 1, clip_step)
                    for i in start_indices:
                        clips.append(frames[i : i + nframes])

                count += 1
                if count % 1000 == 0:
                    print(f"已扫描 {count} 个文件夹，当前内存使用: {os.popen('ps -p %d -o rss=' % os.getpid()).read().strip()} KB")
    
    print(f"扫描完成，共找到 {len(videos)} 个视频片段。")
    return clips, videos

def load_and_transform_frames(frame_list, loader, img_transform=None):
    assert(isinstance(frame_list, list))
    clip = []
    labels = []
    for frame in frame_list:
        
        if isinstance(frame, tuple):
            fpath, label = frame
        elif isinstance(frame, dict):
            fpath = frame["img_path"]
            label = {
                "class_index": frame["class_index"],
                "class_name": frame["class_name"],
                "class_caption": frame["class_caption"],
                }
        
        labels.append(label)
        img = loader(fpath)
        if img_transform is not None:
            img = img_transform(img)
        img = img.view(img.size(0),1, img.size(1), img.size(2))
        clip.append(img)
    return clip, labels[0] # all frames have same label..

def frame_path_from_item(frame):
    if isinstance(frame, dict):
        return frame["img_path"]
    return frame[0]


def render_building_condition_frames(frame_list, condition_root, img_transform=None):
    first_frame = frame_list[0]
    first_path = frame_path_from_item(first_frame)

    clip_name = os.path.basename(os.path.dirname(first_path))
    condition_id = clip_name.split('_', 1)[0]
    condition_path = os.path.join(condition_root, f"{condition_id}.json")
    with open(condition_path, "r") as f:
        buildings = json.load(f)

    cond_frames = []
    for i, frame in enumerate(frame_list):
        frame_path = frame_path_from_item(frame)

        height_idx = height_index_from_frame_name(frame_path, i + 1)
        img = Image.new("L", (256, 256), 0)
        draw = ImageDraw.Draw(img)
        for polygon, height_values in buildings:
            height = float(height_values[0])
            if height < height_idx:
                continue
            draw.polygon([tuple(point) for point in polygon], fill=255)

        img = img.convert("L").convert("RGB")
        if img_transform is not None:
            img = img_transform(img)
        img = img[:1].view(1, 1, img.size(1), img.size(2))
        cond_frames.append(img)

    return torch.cat(cond_frames, 1)


def render_view_condition_frames(frame_list, view_condition_root, img_transform=None):
    first_path = frame_path_from_item(frame_list[0])
    clip_name = os.path.basename(os.path.dirname(first_path))
    view_dir = os.path.join(view_condition_root, clip_name)
    if not os.path.isdir(view_dir):
        raise FileNotFoundError(f"Missing view condition directory: {view_dir}")

    cond_frames = []
    for i, frame in enumerate(frame_list):
        frame_path = frame_path_from_item(frame)
        height_idx = height_index_from_frame_name(frame_path, i + 1)
        path = os.path.join(view_dir, f"h{height_idx}.png")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing view condition frame: {path}")
        img = Image.open(path).convert("L").convert("RGB")
        if img_transform is not None:
            img = img_transform(img)
        img = img[:1].view(1, 1, img.size(1), img.size(2))
        cond_frames.append(img)

    return torch.cat(cond_frames, 1)

class VideoFrameDataset(data.Dataset):
    def __init__(self,
        data_root,
        resolution,
        video_length,                   # clip length
        dataset_name="",
        subset_split="",
        annotation_dir=None,
        spatial_transform="",
        temporal_transform="",
        frame_stride=1,
        clip_step=None,
        condition_root=None,
        view_condition_root=None,
        condition_key="condition",
        ):
        
        self.loader = default_loader
        self.video_length = video_length
        self.subset_split = subset_split
        self.temporal_transform = temporal_transform
        self.spatial_transform = spatial_transform
        self.frame_stride = frame_stride
        self.dataset_name = dataset_name
        self.condition_root = condition_root
        self.view_condition_root = view_condition_root
        self.condition_key = condition_key

        assert(subset_split in ["train", "test", "all", ""]) # "" means no subset_split directory.
        assert(self.temporal_transform in ["", "rand_clips"])

        if subset_split == 'all':
            video_dir = os.path.join(data_root, "train")
        else:
            video_dir = os.path.join(data_root, subset_split)
        
        if dataset_name == 'UCF-101':
            if annotation_dir is None:
                annotation_dir = os.path.join(data_root, "ucfTrainTestlist")
            class_to_idx = class_name_to_idx(annotation_dir)
            assert(len(class_to_idx) == 101), f'num of classes = {len(class_to_idx)}, not 101'
        elif dataset_name == 'sky':
            classes, class_to_idx = find_classes(video_dir)
        else:
            class_to_idx = None
            classes, class_to_idx = find_classes_rm(video_dir)
            print("success!")
        
        # make dataset
        if dataset_name == 'UCF-101' or 'rm':
            print("rm")
            func = make_dataset_ucf
        else:
            func = make_dataset
        self.clips, self.videos = func(video_dir, video_length,  class_to_idx, frame_stride=frame_stride, clip_step=clip_step)
        assert(len(self.clips[0]) == video_length), f"Invalid clip length = {len(self.clips[0])}"
        if self.temporal_transform == 'rand_clips':
            self.clips = self.videos
        
        if subset_split == 'all':
            # add test videos
            video_dir = video_dir.rstrip('/train')+'/test'
            cs, vs = func(video_dir, video_length, class_to_idx)
            if self.temporal_transform == 'rand_clips':
                self.clips += vs
            else:
                self.clips += cs
        
        print('[VideoFrameDataset] number of videos:', len(self.videos))
        print('[VideoFrameDataset] number of clips', len(self.clips))

        # check data
        if len(self.clips) == 0:
            raise(RuntimeError(f"Found 0 clips in {video_dir}. \n"
                               "Supported image extensions are: " + 
                               ",".join(IMG_EXTENSIONS)))
        
        # data transform
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if self.spatial_transform == "center_crop_resize":
            print('Spatial transform: center crop and then resize')
            self.video_transform = transforms.Compose([
                transforms.Resize(resolution),
                transforms_video.CenterCropVideo(resolution),
                ])
        elif self.spatial_transform == "resize":
            print('Spatial transform: resize with no crop')
            self.video_transform = transforms.Resize((resolution, resolution))
        elif self.spatial_transform == "random_crop":
            self.video_transform = transforms.Compose([
                transforms_video.RandomCropVideo(resolution),
                ])
        elif self.spatial_transform == "":
            self.video_transform = None
        else:
            raise NotImplementedError

    def __getitem__(self, index):
        # get clip info
        if self.temporal_transform == 'rand_clips':
            raw_video = self.clips[index]
            rand_idx = random.randint(0, len(raw_video) - self.video_length)
            clip = raw_video[rand_idx:rand_idx+self.video_length]
        else:
            clip = self.clips[index]
        assert(len(clip) == self.video_length), f'current clip_length={len(clip)}, target clip_length={self.video_length}, {clip}'
        
        # make clip tensor
        frames, labels = load_and_transform_frames(clip, self.loader, self.img_transform)
        assert(len(frames) == self.video_length), f'current clip_length={len(frames)}, target clip_length={self.video_length}, {clip}'
        frames = torch.cat(frames, 1) # c,t,h,w
        condition = None
        if self.condition_root is not None:
            condition = render_building_condition_frames(clip, self.condition_root, self.img_transform)
            if self.view_condition_root is not None:
                view_condition = render_view_condition_frames(clip, self.view_condition_root, self.img_transform)
                condition = torch.cat([condition, view_condition], dim=0)
        if self.video_transform is not None:
            frames = self.video_transform(frames)
            if condition is not None:
                condition = self.video_transform(condition)
        
        example = dict()
        example["image"] = frames
        if condition is not None:
            example[self.condition_key] = condition
        if labels is not None and self.dataset_name == 'UCF-101':
            example["caption"] = labels["class_caption"]
            example["class_label"] = labels["class_index"]
            example["class_name"] = labels["class_name"]
        example["frame_stride"] = self.frame_stride
        return example

    def __len__(self):
        return len(self.clips)
