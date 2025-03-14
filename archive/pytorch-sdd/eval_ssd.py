import torch
from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.utils import box_utils, measurements
from vision.utils.misc import str2bool
import argparse
import pathlib
import numpy as np
import logging
import sys
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor


class MeanAPEvaluator:
    """
    Mean Average Precision (mAP) evaluator
    """
    def __init__(self, dataset, net, arch='mb1-ssd', eval_dir='models/eval_results', 
                 nms_method='hard', iou_threshold=0.5, use_2007_metric=True, device='cuda:0'):
                 
        self.dataset = dataset
        self.net = net
        self.iou_threshold = iou_threshold
        self.use_2007_metric = use_2007_metric

        self.eval_path = pathlib.Path(eval_dir)
        self.eval_path.mkdir(exist_ok=True)
    
        self.true_case_stat, self.all_gb_boxes, self.all_difficult_cases = self.group_annotation_by_class(self.dataset)
        
        if arch == 'vgg16-ssd':
            self.predictor = create_vgg_ssd_predictor(net, nms_method=nms_method, device=device)
        elif arch == 'mb1-ssd':
            self.predictor = create_mobilenetv1_ssd_predictor(net, nms_method=nms_method, device=device)
        elif arch == 'mb1-ssd-lite':
            self.predictor = create_mobilenetv1_ssd_lite_predictor(net, nms_method=nms_method, device=device)
        elif arch == 'sq-ssd-lite':
            self.predictor = create_squeezenet_ssd_lite_predictor(net,nms_method=nms_method, device=device)
        elif arch == 'mb2-ssd-lite':
            self.predictor = create_mobilenetv2_ssd_lite_predictor(net, nms_method=nms_method, device=device)
        else:
            raise ValueError(f"Invalid network architecture type '{arch}' - it should be one of:  vgg16-ssd, mb1-ssd, mb1-ssd-lite, mb2-ssd-lite, sq-ssd-lite")

    def compute(self):
        is_test = self.net.is_test
        self.net.is_test = True
        
        results = []

        for i in range(len(self.dataset)):
            logging.debug(f"evaluating average precision   image {i} / {len(self.dataset)}")
            image = self.dataset.get_image(i)
            boxes, labels, probs = self.predictor.predict(image)
            indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
            results.append(torch.cat([
                indexes.reshape(-1, 1),
                labels.reshape(-1, 1).float(),
                probs.reshape(-1, 1),
                boxes + 1.0  # matlab's indexes start from 1
            ], dim=1))
            
        results = torch.cat(results)
        self.net.is_test = is_test
        
        for class_index, class_name in enumerate(self.dataset.class_names):
            if class_index == 0: continue  # ignore background
            prediction_path = self.eval_path / f"det_test_{class_name}.txt"
            with open(prediction_path, "w") as f:
                sub = results[results[:, 1] == class_index, :]
                for i in range(sub.size(0)):
                    prob_box = sub[i, 2:].numpy()
                    image_id = self.dataset.ids[int(sub[i, 0])]
                    print(
                        image_id + "\t" + " ".join([str(v) for v in prob_box]).replace(" ", "\t"),
                        file=f
                    )
        aps = []
        
        for class_index, class_name in enumerate(self.dataset.class_names):
            if class_index == 0:
                continue
            prediction_path = self.eval_path / f"det_test_{class_name}.txt"
            ap = self.compute_average_precision_per_class(
                self.true_case_stat[class_index],
                self.all_gb_boxes[class_index],
                self.all_difficult_cases[class_index],
                prediction_path,
                self.iou_threshold,
                self.use_2007_metric
            )
            aps.append(ap)

        return sum(aps)/len(aps), aps
      
    def log_results(self, mean_ap, class_ap, prefix=''):
        logging.info(f"{prefix}Average Precision Per-class:")
        
        for i in range(len(class_ap)):
            logging.info(f"    {self.dataset.class_names[i+1]}: {class_ap[i]}")
            
        logging.info(f"{prefix}Mean Average Precision (mAP):  {mean_ap}")
        
    def group_annotation_by_class(self, dataset):
        true_case_stat = {}
        all_gt_boxes = {}
        all_difficult_cases = {}
        for i in range(len(dataset)):
            image_id, annotation = dataset.get_annotation(i)
            gt_boxes, classes, is_difficult = annotation
            gt_boxes = torch.from_numpy(gt_boxes)
            for i, difficult in enumerate(is_difficult):
                class_index = int(classes[i])
                gt_box = gt_boxes[i]
                if not difficult:
                    true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

                if class_index not in all_gt_boxes:
                    all_gt_boxes[class_index] = {}
                if image_id not in all_gt_boxes[class_index]:
                    all_gt_boxes[class_index][image_id] = []
                all_gt_boxes[class_index][image_id].append(gt_box)
                if class_index not in all_difficult_cases:
                    all_difficult_cases[class_index]={}
                if image_id not in all_difficult_cases[class_index]:
                    all_difficult_cases[class_index][image_id] = []
                all_difficult_cases[class_index][image_id].append(difficult)

        for class_index in all_gt_boxes:
            for image_id in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
        for class_index in all_difficult_cases:
            for image_id in all_difficult_cases[class_index]:
                all_gt_boxes[class_index][image_id] = all_gt_boxes[class_index][image_id].clone().detach() #torch.tensor(all_gt_boxes[class_index][image_id])
        return true_case_stat, all_gt_boxes, all_difficult_cases


    def compute_average_precision_per_class(self, num_true_cases, gt_boxes, difficult_cases,
                                            prediction_file, iou_threshold, use_2007_metric):
        with open(prediction_file) as f:
            image_ids = []
            boxes = []
            scores = []
            for line in f:
                t = line.rstrip().split("\t")
                image_ids.append(t[0])
                scores.append(float(t[1]))
                box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
                box -= 1.0  # convert to python format where indexes start from 0
                boxes.append(box)
            scores = np.array(scores)
            sorted_indexes = np.argsort(-scores)
            boxes = [boxes[i] for i in sorted_indexes]
            image_ids = [image_ids[i] for i in sorted_indexes]
            true_positive = np.zeros(len(image_ids))
            false_positive = np.zeros(len(image_ids))
            matched = set()
            for i, image_id in enumerate(image_ids):
                box = boxes[i]
                if image_id not in gt_boxes:
                    false_positive[i] = 1
                    continue

                gt_box = gt_boxes[image_id]
                ious = box_utils.iou_of(box, gt_box)
                max_iou = torch.max(ious).item()
                max_arg = torch.argmax(ious).item()
                if max_iou > iou_threshold:
                    if difficult_cases[image_id][max_arg] == 0:
                        if (image_id, max_arg) not in matched:
                            true_positive[i] = 1
                            matched.add((image_id, max_arg))
                        else:
                            false_positive[i] = 1
                else:
                    false_positive[i] = 1

        true_positive = true_positive.cumsum()
        false_positive = false_positive.cumsum()
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / num_true_cases
        if use_2007_metric:
            return measurements.compute_voc2007_average_precision(precision, recall)
        else:
            return measurements.compute_average_precision(precision, recall)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
    
    parser.add_argument('--net', default="vgg16-ssd", help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
    parser.add_argument("--model", type=str, help="Path to the trained PyTorch checkpoint")
    parser.add_argument("--dataset_type", default="voc", type=str, help="Specify dataset type. Currently support voc and open_images.")
    parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
    parser.add_argument("--label_file", type=str, help="The label file path.")
    parser.add_argument("--use_cuda", type=str2bool, default=True)
    parser.add_argument("--use_2007_metric", type=str2bool, default=True)
    parser.add_argument("--nms_method", type=str, default="hard")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
    parser.add_argument("--eval_dir", default="models/eval_results", type=str, help="The directory to store evaluation results.")
    parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                        help='Width Multiplifier for MobilenetV2')
                        
    args = parser.parse_args()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
                    
    # load the dataset
    if args.dataset_type == "voc":
        dataset = VOCDataset(args.dataset, is_test=True)
    elif args.dataset_type == 'open_images':
        dataset = OpenImagesDataset(args.dataset, dataset_type="test")

    # create the network
    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(dataset.class_names), is_test=True)
    elif args.net == 'mb1-ssd':
        net = create_mobilenetv1_ssd(len(dataset.class_names), is_test=True)
    elif args.net == 'mb1-ssd-lite':
        net = create_mobilenetv1_ssd_lite(len(dataset.class_names), is_test=True)
    elif args.net == 'sq-ssd-lite':
        net = create_squeezenet_ssd_lite(len(dataset.class_names), is_test=True)
    elif args.net == 'mb2-ssd-lite':
        net = create_mobilenetv2_ssd_lite(len(dataset.class_names), width_mult=args.mb2_width_mult, is_test=True)
    else:
        logging.fatal(f"Invalid network architecture type '{arch}' - it should be one of:  vgg16-ssd, mb1-ssd, mb1-ssd-lite, mb2-ssd-lite, sq-ssd-lite")
        parser.print_help(sys.stderr)
        sys.exit(1)  

    # load the model
    logging.info(f"loading model {args.model}")
    net.load(args.model)
    net = net.to(DEVICE)
    logging.info(f"loaded model {args.model}")
    
    # eval the mAP
    eval = MeanAPEvaluator(dataset, net, arch=args.net, eval_dir=args.eval_dir, 
                           nms_method=args.nms_method, iou_threshold=args.iou_threshold,
                           use_2007_metric=args.use_2007_metric, device=DEVICE)
                                 
    mean_ap, class_ap = eval.compute()
    eval.log_results(mean_ap, class_ap)
