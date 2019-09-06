import torch.utils.data as tdata


class A2DComposition(tdata.Dataset):
	def __init__(self, args, transform=None, mode='train'):
		self.args=args
		self.transform=transform
		self.mode=mode

		self.actions, self.actors, self.pairs, self.train_pairs, self.test_pairs = self.parse_split()
		self.train_data, self.test_data = self.get_split_info()
		self.data = self.train_data if self.phase=='train' else self.test_data

		self.action2idx = {action: idx for idx, action in enumerate(self.actions)}
		self.actor2idx = {actor: idx for idx, actor in enumerate(self.actors)}
		self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

		self.actor_affordance = {}
		self.train_actor_affordance = {}
		for _actor in self.actors:
			candidates = [action for (_, _,action, actor) in self.train_data+self.test_data if actor==_actor]
			self.actor_affordance[_actor] = list(set(candidates))

			candidates = [actor for (_, _,action, actor) in self.train_data if actor==_actor]
			self.train_actor_affordance[_actor] = list(set(candidates))

		feat_file = None # pre-joint features path
		activation_data = torch.load(feat_file) # [features,names] features=>(N,d)
		self.activations = dict(zip(activation_data['files'], activation_data['features']))
		self.feat_dim = activation_data['features'].size(1) # d

	def parse_split(self):
		##################
		tr_actions, tr_actors, tr_pairs=None
		ts_actions, ts_actors, ts_pairs=None
		##################

		all_actions, all_actors =  sorted(list(set(tr_actions+ts_actions))), sorted(list(set(tr_actors+ts_actors)))    
		all_pairs = sorted(list(set(tr_pairs + ts_pairs)))

		return all_actions, all_actors, all_pairs, tr_pairs, ts_pairs

	def get_split_info(self):
		# [img_path, [t img_path], actor, action]
		##################
		pass
		##################

		return train_data, test_data