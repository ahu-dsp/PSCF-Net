import argparse
parser = argparse.ArgumentParser(description='arg')

parser.add_argument('--lr', type=int , default=0.00001)
parser.add_argument('--BatchSize', type=int, default= 8, help='batch size')
parser.add_argument('--epoch', type=int , default = 150 )
parser.add_argument('--train', type=bool, default = True )
parser.add_argument('--sensor', type=str, default='IKONOS')
parser.add_argument('--filepath', type=str, default='train/IKONOS')
parser.add_argument('--pre_train', type=bool, default= False )
parser.add_argument('--pretrain_model', type=str, default='/Share/home/Z21301095/test/model/SmoothL1Loss+50ssim_cfb=3--save270999--lr=1e-05.pth')
parser.add_argument('--testmodel', type=str, default='model/IKONOS_CFB=4_save358999--lr=1e-05.pth')
parser.add_argument('--testtype', type=str, default='NO-REFERENCE METRICS1')

parser.add_argument('--cuda', action='store_true', help='use cuda?')
opt = parser.parse_args()