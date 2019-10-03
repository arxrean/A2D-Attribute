import argparse

def p_parse():
	parser = argparse.ArgumentParser()
	parser.add_argument("--session_name", default="a2d", type=str)
	# data
	parser.add_argument(
		"--a2d_root", default='E:/A2D/Release', type=str)
		#"--a2d_root", default='/u/zkou2/Data/A2D/Release', type=str)
	parser.add_argument('--csv_path', default='./repo/newVideoSet.csv')
	parser.add_argument('--t', type=int, default=0)
	parser.add_argument('--input_size', type=int, default=224)
	parser.add_argument('--class_num', type=int, default=43)
	parser.add_argument('--actor_num', type=int, default=7)
	parser.add_argument('--action_num', type=int, default=9)
	# config
	parser.add_argument("--num_workers", default=0, type=int)
	parser.add_argument("--batch_size", default=512, type=int)
	parser.add_argument("--max_epoches", default=2000, type=int)
	parser.add_argument("--cuda", default=False, type=bool)
	parser.add_argument("--pretrained", default=True, type=bool)
	parser.add_argument("--use_feat", default=True, type=bool)
	# train
	parser.add_argument("--lr", default=1e-5, type=float)
	parser.add_argument("--lr_dec", default=1.0, type=float)
	parser.add_argument("--wt_dec", default=5e-4, type=float)
	parser.add_argument("--op_img_dim", default=300, type=int)
	parser.add_argument("--triplet_margin", default=5.0, type=float)
	parser.add_argument("--constraint_cls", default=0, type=float)
	# save
	parser.add_argument("--save_root", default='./save/', type=str)
	# feature
	parser.add_argument("--feature_path",default='./repo/joint_img_feature.npy')

	args = parser.parse_args()

	return args
