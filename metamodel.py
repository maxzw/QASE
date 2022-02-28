import torch

from gcn import GCNModel

class MetaModel(torch.nn.Module):
    def __init__(
        self,
        num_buckets:    int,    # number of buckets
        num_planes:     int,    # numbef of hyperplanes (GCNModels) per bucket
        gcn_layers:     int,    # number of layers per GCNModel
        gcn_readout:    str,    # GCNModel readout method (choose from ['max', 'sum', 'TM'])
        embed_dims:     int,    # embedding dimensions
        num_rels:       int,    # number of relations
        ):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_planes = num_planes
        self.gcn_layers = gcn_layers
        self.gcn_readout = gcn_readout
        self.embed_dims = embed_dims
        self.num_rels = num_rels

        self.bucket_size = self.num_planes * self.embed_dims

        # initiate GCNModels
        self.submodels = [
            GCNModel(
                self.embed_dims,
                self.gcn_layers,
                self.num_rels,
                self.gcn_readout
                ) for _ in range(self.num_buckets * self.num_planes)
        ]


        def forward(self, x):
            """
            Performs forward pass for all submodels and combines their output.
            """
            # GCNModel output is matrix with hyperplanes in size [batchsize * embed_dim],
            # we concatenate these along the first axis.
            hyp_list = [gcn(x) for gcn in self.submodels]
            hyp_matrix = torch.cat((hyp_list), 1)
            return hyp_matrix


        def bucket_loss(self, hyp_matrix, answers):
            """
            Calculates the loss of a single bucket based on hyperplanes
            and the correct answer.
            """
            loss = 0
            return loss


        def contains_answer(self, hyp_matrix, answers):
            """
            Returns indicator for if the bucket contains the answer set.
            """
            pass


        def calculate_loss(self, hyp_matrix, answers):
            # get buckets: list of Tensors [batchsize, embed_dim * num_planes], size = num_buckets
            buckets = torch.chunk(hyp_matrix, self.num_buckets, dim=1)
            
            # calculate proxy loss per bucket
            proxy_losses = [bucket_loss(bucket, answers) for bucket in buckets]

            # delegate loss for each bucket
            loss_indicator = [contains_answer(bucket, answers) for bucket in buckets]

            # if no buckets contain correct answers: all buckets included in backward pass
            # if one or more buckets contain correct answers: only those buckets included in backward pass
            loss = 0
            for pl, ind in zip(proxy_losses, loss_indicator):
                if ind:
                    loss += pl
            return loss