from loss import AnswerSpaceLoss
from model import MetaModel
from loader import *
from train import train
    
if __name__ == "__main__":

    data_dir = "./data/AIFB/processed/"
    embed_dim = 128
    num_epochs = 6
    eval_freq = 2

    model = MetaModel(
        data_dir,
        embed_dim,
        gcn_layers=3
        )
        
    loss_fn = AnswerSpaceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    train_queries, train_info = get_queries(data_dir, split='train')
    val_queries, val_info = get_queries(data_dir, split='val')
    
    train_dataloader = get_dataloader(
        train_queries,
        batch_size = 50,
        shuffle=True,
        num_workers=1
        )

    val_dataloader = get_dataloader(
        train_queries,
        batch_size = 50,
        shuffle=False,
        num_workers=1
        )
    
    training_report = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=num_epochs,
        eval_freq=eval_freq
    )
    
    print(training_report)