import os
import json


"""
For pre_processing on test videos, and get txt file

1. get each video_path from 'test_video_info.json', and obtain the path list
2. loop through the path liat, and put the pictures into SportsMOT
3. get each video's MOT result, and save in each video file
4. load the txt file, and get each video's player sequences
5. save the video_player_bbox_info.json
    {"video10000": {"1": [[x,x,x,x], [0,0,0,,], [x,x,x,x]], "2": [[x,x,x,x], [0,0,0,,], [x,x,x,x]], ......  }}
"""

import argparse
import os
import os.path as osp
import time
import cv2
import torch

from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.byte_tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")

    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("--name", type=str, default="yolox-x", help="model name")

    parser.add_argument(
        "--video_path",
        default="",
        type=str,
        help="video path",
    )
    parser.add_argument("--video_id", type=str, default="", help="video name")
    parser.add_argument(
        "--Out_vis_path",
        default="/PATH/TO/VG_NBA_MOT_VIS",
        type=str,
        help="vis results, and you can check each vis file",
    )
# images
    parser.add_argument(
        "--demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument(
        "--path", default="/PATH/TO/Images/7", help="path to images or video"
        #"--path", default="/PATH/TO/videos/palace.mp4", help="path to images or video"
    )
# video
#     parser.add_argument(
#         "--demo", default="webcam", help="demo type, eg. image, video and webcam"
#     )
#     parser.add_argument(
#         "--path", default="/PATH/TO/MixSort/Images/video-1/32.mp4",
#         help="path to images or video"
#         # "--path", default="/PATH/TO/MixSort/videos/palace.mp4", help="path to images or video"
#     )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        default=True,
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "--exp_file",
        default="/PATH/TO/yolox_x_sportsmot.py",
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("--ckpt", default="/PATH/TO/MixFormer_sports_train.pth.tar", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.01, type=float, help="test conf")
    parser.add_argument("--nms", default=0.7, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=5, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.7, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                f.write(line)
    logger.info('save results to {}'.format(filename))


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        with torch.no_grad():
            #timer.tic()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info


def image_demo(predictor, vis_folder, current_time, video_path, video_id, args):
    if osp.isdir(video_path):
        files = get_image_list(video_path)
    else:
        files = [video_path]
    files.sort()
    tracker = BYTETracker(args, frame_rate=args.fps)
    timer = Timer()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        #print("img_info: ", img_info)
        #print("outputs: ", outputs[0])
        if outputs[0] is not None:
            # print("exp.text_size: ", exp.test_size)  # exp.text_size:  (800, 1440)
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
            #print("online_targets: ", online_targets)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                #print("t: ", t)
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    # save results
                    results.append(
                        f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )
            #print("online_ids: ", online_ids)
            #print("online_tlwhs: ", online_tlwhs)
            timer.toc()
            online_im = plot_tracking(
                img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            timer.toc()
            online_im = img_info['raw_img']

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            #timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            save_folder = osp.join(vis_folder, video_id)
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        # if frame_id % 20 == 0:
        #     logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(video_path, f"MOT.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")
        time.sleep(1)


    # load txt file and analyse, get the json file



def Track_MOT(exp, video_path, video_id, args):
    print("video_path: ", video_path)
    print("video_id: ", video_id)
    # if not args.experiment_name:
    #     args.experiment_name = exp.exp_name
    #
    # output_dir = osp.join(exp.output_dir, args.experiment_name)
    # os.makedirs(output_dir, exist_ok=True)
    output_dir = args.Out_vis_path

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_vis")
        print("vis_folder: ", vis_folder)
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)
    current_time = time.localtime()
    if args.demo == "image":
        # with open(args.text_video_info,
        #           encoding='utf-8') as Video_f:
        #     result_info = json.load(Video_f)
        # for video_id, video_info in result_info.items():
        #     video_path = video_info["save_path"]
        image_demo(predictor, vis_folder, current_time, video_path, video_id, args)



if __name__ == "__main__":
    #args = make_parser().parse_args()

    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    Track_MOT(exp, args.video_path, args.video_id, args)
    print("-----------------------------------")
