import torch.nn as nn
import torchvision
from torchvision.models import resnet50
import torch


class Group_Activity_Temporal_Classifier(nn.Module):
    def __init__(self, person_classifier, num_classes=8,input_size=2048, hidden_size=256, num_layers=1):
        super(Group_Activity_Temporal_Classifier, self).__init__()

        self.person_feature_extractor_fc_in = person_classifier.in_features
        print(f"Person feature extractor output dimension: {self.person_feature_extractor_fc_in}")

        layers = list(person_classifier.backbone.children())[:-1]
        self.person_feature_extractor = nn.Sequential(*layers)  # remove last FC layer

        for param in self.person_feature_extractor.parameters():  # Feature extractor is frozen
            param.requires_grad = False

        self.lstm = nn.LSTM( # input (seq,frames,2048) out (seq,frames,hidden_size)
                            input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout = 0.5 if num_layers > 1 else 0.0
                            )

        self.fc =  nn.Sequential(
            nn.Linear(in_features= input_size+hidden_size,out_features= 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features= 512,out_features= 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features= 128, out_features= num_classes),
        )

    def forward(self,x):
        """
        Input x shape: [Batch, Num_Players, Time_Steps, C, H, W]
        """
        batch,player,frame, c,h,w = x.shape

        # Merge All for CNN
        x = x.view(batch*player*frame,c,h,w) # (seq * 12 * 9, 3, 244, 244)

        # CNN Feature Extraction
        features= self.feature_extractor(x) # (seq * 12 * 9, 2048, 1, 1)
        features = features.view(batch * player * frame, -1) # (seq * 12 * 9, 2048)

        # Return to original dimension
        features = features.view(batch,player,frame,-1) # (batch,12,9,2048)

        # Pool across the Players dimension
        pooled_features = torch.max(features, dim=1)[0]  # (batch,9,2048)

        # LSTM over time for each player
        lstm_out, (hidden,cell) = self.lstm(pooled_features) # (batch,9,hidden_size)

        # Grab the final temporal state & last spatial feature for each player and combine them
        final_lstm_out = lstm_out[:, -1, :]  # (batch, Hidden) (take the last frame only)
        last_cnn_feature = pooled_features[:, -1, :] # (batch, 2048)
        combined_features = torch.cat([final_lstm_out,last_cnn_feature],dim=1) # (batch,hidden_size+2048)

        # Group Classification
        out = self.fc(combined_features)
        return out
