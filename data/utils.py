import numpy as np, pandas as pd
import os
import vector
import sklearn
import joblib
from torch import nn

def convert_lorentz_vector(data):
	"""
	Take data as a numpy array of shape (n_events, 4) of 4 momenta, (E, px, py, pz) 
	and convert to scikit-hep vector for manipulation
	"""
	data = vector.arr({
		'E': data[:, 0],
		'px': data[:, 1],
		'py': data[:, 2],
		'pz': data[:, 3],
	})
	return data

def sort_particle(array, sort_by):
	"""
	Take data as numpy array of shape (n_events, n_particles, n_variables) and sort 
	the particles in each event by the variable at position given by sort_by
	"""
	data = array.copy()
	indices = np.argsort(data[:,:, sort_by], axis=-1)
	for pos, idx in enumerate(indices):
		data[pos] = data[pos][idx]
	return data

def shuffle(array: np.ndarray):
	from numpy.random import MT19937
	from numpy.random import RandomState, SeedSequence
	np_rs = RandomState(MT19937(SeedSequence(123456789)))
	np_rs.shuffle(array)

def read_dataframe(filename: str or list, sep=",", engine=None, prog_bar=False):
	if not isinstance(filename, list):
		filename = [filename]
	if prog_bar: 
		from tqdm import tqdm
		filename = tqdm(filename)
	df_list = []
	for f in filename:
		df_list.append(
			pd.read_csv(f, sep=sep, header=None, names=None, engine=engine)
		)
	df = pd.concat(df_list, ignore_index=True)
	return df

def read_data( filename, train_particle_type=False, save_encoder=None, read_encoder=None, max_etot=None, min_etot=None, prog_bar =False ):
	if not isinstance(filename, list): filename=[filename]
	if prog_bar:
		from tqdm import tqdm
		filename = tqdm(filename)
	p12s, m12s, outcomes = [], [], []
	for f in filename:
		df = read_dataframe(f, sep=' ')
		mask = (~ df.isna().to_numpy()).all(1)
		data = df.to_numpy()[:, :-5]
		# transformations = df.to_numpy()[:, -5:]
		# data, transformations = sklearn.utils.shuffle(data, transformations, random_state=0)
		data = data.reshape((data.shape[0], -1, 5))

		# if train_particle_type:
		# 	label_encoder = ExtendedLabelEncoder()
		# 	particle_types = data[:, :, 0:1]
		# 	label_encoder.fit(particle_types)
		# 	if isinstance(read_encoder, str):
		# 		scaler = joblib.load(read_encoder)
		# 	if isinstance(save_encoder, str):
		# 		os.makedirs(os.path.dirname(save_encoder), exist_ok=True)
		# 		joblib.dump(label_encoder, save_encoder) 
		# 	particle_types = label_encoder.transform(particle_types).reshape( (data.shape[0], -1, 1) )
		# 	data = np.concatenate( [data, particle_types] , axis=-1)
			# print(data.shape)
		
		data=data[:,:, 1:]

		condition = data[:, :2]
		outcome = data[:, 2:]

		p1 = convert_lorentz_vector(condition[:, 0, :].copy())
		p2 = convert_lorentz_vector(condition[:, 1, :].copy())
		p12 = (p1+p2).tau.to_numpy()[:, np.newaxis]
		# mask = np.array([True]*p12.shape[0])
		if isinstance(max_etot, (int, float)):
			mask *= (p1+p2).tau.to_numpy() < max_etot 
		if isinstance(min_etot, (float, int)):
			mask *= min_etot < (p1+p2).tau.to_numpy()
		p1, p2, p12, outcome = p1[mask], p2[mask], p12[mask], outcome[mask]

		m1=p1.mass.to_numpy()[:, np.newaxis]
		m2=p2.mass.to_numpy()[:, np.newaxis]
		m12=m1+m2
		p12s.append(p12)
		m12s.append(m12)
		outcomes.append(outcome)

	return sklearn.utils.shuffle(np.concatenate(p12s, axis=0), np.concatenate(m12s, axis=0), np.concatenate(outcomes, axis=0), random_state=42)

def make_mlp(
	input_size,
	sizes,
	hidden_activation="ReLU",
	output_activation="ReLU",
	dropout_rate=0.,
	batch_norm=True,
):
	"""Construct an MLP with specified fully-connected layers."""
	hidden_activation = getattr(nn, hidden_activation)
	if output_activation is not None:
		output_activation = getattr(nn, output_activation)
	layers = []
	n_layers = len(sizes)
	sizes = [input_size] + sizes
	# Hidden layers
	for i in range(n_layers - 1):
		layers.append(nn.Linear(sizes[i], sizes[i + 1]) if sizes[i] > 0 else nn.LazyLinear(sizes[i+1]) )
		layers.append(nn.Dropout1d(dropout_rate))
		if batch_norm:
			layers.append(nn.BatchNorm1d(sizes[i + 1], track_running_stats=False, affine=False))
		layers.append(hidden_activation())
	# Final layer
	layers.append(nn.Linear(sizes[-2], sizes[-1]))
	if output_activation is not None:
		if batch_norm:
			layers.append(nn.BatchNorm1d(sizes[-1], track_running_stats=False, affine=False))
		layers.append(output_activation())
	return nn.Sequential(*layers)