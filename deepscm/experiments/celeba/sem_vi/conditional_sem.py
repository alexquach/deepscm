import torch
import pyro

from pyro.nn import pyro_method
from pyro.distributions import Normal, Bernoulli, TransformedDistribution
from pyro.distributions.conditional import ConditionalTransformedDistribution
from deepscm.distributions.transforms.affine import ConditionalAffineTransform
from pyro.nn import DenseNN

from deepscm.experiments.celeba.sem_vi.base_sem_experiment import BaseVISEM, MODEL_REGISTRY


class ConditionalVISEM(BaseVISEM):
    context_dim = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # facial_hair flow
        facial_hair_net = DenseNN(1, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.facial_hair_flow_components = ConditionalAffineTransform(context_nn=facial_hair_net, event_dim=0)
        self.facial_hair_flow_transforms = [self.facial_hair_flow_components, self.facial_hair_flow_constraint_transforms]

        embedding_net = DenseNN(2, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        self.embedding_flow_components = ConditionalAffineTransform(context_nn=embedding_net, event_dim=0)
        self.embedding_flow_transforms = [self.embedding_flow_components, self.embedding_flow_constraint_transforms]

        # # ventricle_volume flow
        # ventricle_volume_net = DenseNN(2, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        # self.ventricle_volume_flow_components = ConditionalAffineTransform(context_nn=ventricle_volume_net, event_dim=0)
        # self.ventricle_volume_flow_transforms = [self.ventricle_volume_flow_components, self.ventricle_volume_flow_constraint_transforms]

        # # brain_volume flow
        # brain_volume_net = DenseNN(2, [8, 16], param_dims=[1, 1], nonlinearity=torch.nn.LeakyReLU(.1))
        # self.brain_volume_flow_components = ConditionalAffineTransform(context_nn=brain_volume_net, event_dim=0)
        # self.brain_volume_flow_transforms = [self.brain_volume_flow_components, self.brain_volume_flow_constraint_transforms]

    @pyro_method
    def pgm_model(self):
        sex_dist = Bernoulli(logits=self.sex_logits).to_event(1)

        _ = self.sex_logits

        sex = pyro.sample('sex', sex_dist)

        # age_base_dist = Normal(self.age_base_loc, self.age_base_scale).to_event(1)
        # age_dist = TransformedDistribution(age_base_dist, self.age_flow_transforms)

        # age = pyro.sample('age', age_dist)
        # age_ = self.age_flow_constraint_transforms.inv(age)
        # # pseudo call to thickness_flow_transforms to register with pyro
        # _ = self.age_flow_components

        # brain_context = torch.cat([sex, age_], 1)

        # base = noise?
        facial_hair_base_dist = Normal(self.facial_hair_base_loc, self.facial_hair_base_scale).to_event(1)
        facial_hair_dist = ConditionalTransformedDistribution(facial_hair_base_dist, self.facial_hair_flow_transforms).condition(sex)

        facial_hair = pyro.sample('facial_hair', facial_hair_dist)
        # pseudo call to intensity_flow_transforms to register with pyro
        _ = self.facial_hair_flow_components

        # ventricle_context = torch.cat([age_, brain_volume_], 1)

        # ventricle_volume_base_dist = Normal(self.ventricle_volume_base_loc, self.ventricle_volume_base_scale).to_event(1)
        # ventricle_volume_dist = ConditionalTransformedDistribution(ventricle_volume_base_dist, self.ventricle_volume_flow_transforms).condition(ventricle_context)  # noqa: E501

        # ventricle_volume = pyro.sample('ventricle_volume', ventricle_volume_dist)
        # pseudo call to intensity_flow_transforms to register with pyro
        # _ = self.ventricle_volume_flow_components

        # return age, sex, ventricle_volume, beard
        return sex, facial_hair

    @pyro_method
    def model(self):
        # age, sex, ventricle_volume, brain_volume = self.pgm_model()
        sex, facial_hair = self.pgm_model()

        facial_hair_ = self.facial_hair_flow_constraint_transforms.inv(facial_hair)
        # ventricle_volume_ = self.ventricle_volume_flow_constraint_transforms.inv(ventricle_volume)
        # brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(brain_volume)

        # z = pyro.sample('z', Normal(self.z_loc, self.z_scale).to_event(1))

        # latent = torch.cat([z, ventricle_volume_, brain_volume_], 1)
        # print('sex', sex)
        # print('facial_hair', facial_hair)
        # print('facial_hair_', facial_hair_)

        embedding_context = torch.cat([sex, facial_hair_], 1)
        embedding_base_dist = Normal(self.embedding_base_loc, self.embedding_base_scale).to_event(1)
        embedding_dist = ConditionalTransformedDistribution(embedding_base_dist, self.embedding_flow_transforms).condition(embedding_context)

        # x_dist = self._get_transformed_x_dist(latent)

        # We want [-1, 384]?
        # x = pyro.sample('x', x_dist.to_event(0))
        x = pyro.sample('x', embedding_dist.to_event(0))

        # return x, z, age, sex, ventricle_volume, brain_volume
        return x, sex, facial_hair

    @pyro_method
    # def guide(self, x, age, sex, ventricle_volume, brain_volume):
    def guide(self, x, sex, facial_hair):
        with pyro.plate('observations', x.shape[0]):
            hidden = self.encoder(x)

            facial_hair_ = self.facial_hair_flow_constraint_transforms.inv(facial_hair)
            # brain_volume_ = self.brain_volume_flow_constraint_transforms.inv(brain_volume)

            # print(f"hidden shape: {hidden.shape}")
            # print(f"sex shape: {sex.shape}")
            # print(f"facial_hair_ shape: {facial_hair_.shape}")
            hidden = torch.cat([hidden, sex, facial_hair_], 1)

            latent_dist = self.latent_encoder.predict(hidden)

            z = pyro.sample('z', latent_dist)

        return z


MODEL_REGISTRY[ConditionalVISEM.__name__] = ConditionalVISEM
