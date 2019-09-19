# 2019.9.19 基本模型的训练日志

##ifame
>--lr
0.001
--batch-size
4
--weight-decay
1e-3
--arch
resnet50
--num-segments
10
--representation
iframe
--data-root
/data/sjhu/features
--train-list
/data/sjhu/train_sample.txt
--test-list
/data/sjhu/test_sample.txt
--model-prefix
vcdb_medium_iframe_res50_copy_detection
--lr-steps
50
100
150
--epochs
250
--gpus
2
3

效果是最好准确率为89%

#mv
>具体参数 dropout-0.2 weight_decay=1e-4,batchsize=8,numsegments=10

效果是最好准确率 82%

#residual
>具体参数 dropout-0.2 weight_decay=1e-4,batchsize=8,numsegments=10

效果最好时准确率 85%