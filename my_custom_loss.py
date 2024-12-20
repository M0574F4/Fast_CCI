import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl



class CustomLoss(nn.Module):
    def __init__(self, loss_type="mse", combined_loss_ratio=None, l1_weight=0.0, l2_weight=0.0):
        super(CustomLoss, self).__init__()
        self.loss_type = loss_type

        self.l1_weight=l1_weight
        self.l2_weight=l2_weight
        
        self.combined_loss_ratio = combined_loss_ratio
                
        if loss_type == "mse":
            self.loss_fn = self.simple_l2  # Initialize without arguments
        elif loss_type == "mae":
            self.loss_fn = self.simple_l1
        elif loss_type == "ber_loss":
            self.loss_fn = self.ber_loss
        elif loss_type == "combined_ber_soi":
            self.loss_fn = self.combined_ber_soi
        elif loss_type == "mse_score" or loss_type == "mse_ber_score":
            self.loss_fn = self.soft_truncated_mse
        elif loss_type == "wnet_mse_score":
            self.loss_fn = self.WNet_soft_truncated_mse
    
    def soft_truncated_mse_ber(self, model_output, targets, model_parameters=None):
        # Calculate the Mean Squared Error (MSE) term
        mse = nn.functional.mse_loss(model_output['soi'], targets['soi'], reduction='none')
        mse_db = 10 * torch.log10(mse + 1e-10)
        mse_loss, mse_db = self.custom_smooth_transition(mse_db)
    
        # Calculate the Binary Cross-Entropy (BCE) term for bit errors
        # Ensure targets['soi'] is within [0, 1] and model_output['bits'] is the output of a sigmoid
        bce_loss = 1*nn.functional.binary_cross_entropy(model_output['bits'], targets['bits'] )
    
        # Combine the two loss terms
        # You may want to weight these terms differently depending on their relative importance
    
        return mse_loss, mse_db, bce_loss

    def simple_l1(self, model_output, targets, model_parameters=None):
        mse = nn.functional.mse_loss(model_output['soi'], targets['soi'], reduction='none')
        mse_db = 10 * torch.log10(mse + 1e-10)
        # Calculate the Mean Absolute Error (MAE) term
        mae = nn.functional.l1_loss(model_output['soi'], targets['soi'], reduction='mean')
    
        return mae, mse_db

    def simple_l2(self, model_output, targets, model_parameters=None):
        mse = nn.functional.mse_loss(model_output['soi'], targets['soi'], reduction='none')
        mse_db = 10 * torch.log10(mse + 1e-10)
        mse_mean = nn.functional.mse_loss(model_output['soi'], targets['soi'], reduction='mean')
        mse_db_mean = 10 * torch.log10(mse_mean + 1e-10)
    
        return mse_db_mean, mse_db

    

    def soft_truncated_mse(self, model_output, targets, model_parameters=None, hparam=None):
        # Check if model_output is a dictionary
        if isinstance(model_output, dict):
            mse = nn.functional.mse_loss(model_output['soi'], targets['soi'], reduction='none')
        elif isinstance(model_output, list):
            mse = 0.0
            if hparam.model.model_type == 'AdaptiveMultiExitUNet':
                # print('We know it is AdaptiveMultiExitUNet')
                mse_loss = nn.functional.mse_loss(model_output[0], targets['soi'], reduction='none')
                mse = mse_loss + model_output[1]
            else:
                for i, output in enumerate(model_output):
                    mse_loss = nn.functional.mse_loss(output, targets['soi'], reduction='none')
                    mse += hparam.trainer.exit_weights[i] * mse_loss / sum(hparam.trainer.exit_weights)

        else:
            # Use model_output directly and consider bce_loss as zero
            mse = nn.functional.mse_loss(model_output, targets['soi'], reduction='none')
            bce_loss = 0

        mse_db = 10 * torch.log10(mse + 1e-10)
    
        if self.loss_type == "mse_ber_score":
            mse_loss, mse_db = self.custom_smooth_transition(mse_db)
            
            # Compute bce_loss only if model_output is a dictionary
            if isinstance(model_output, dict):
                bce_loss = (mse_loss < -30) * 1 * nn.functional.binary_cross_entropy(model_output['bits'], targets['bits'])
            return mse_loss + bce_loss, mse_db
        else:
            return self.custom_smooth_transition(mse_db)
    
    
        
    def WNet_soft_truncated_mse(self, model_output, targets, model_parameters=None):
        mse1 = nn.functional.mse_loss(model_output['soi'], targets['soi'], reduction='none')
        mse2 = nn.functional.mse_loss(model_output['interference'], targets['mixture']-targets['soi'], reduction='none')
        
        mse_db = 10 * torch.log10(mse1 + mse2 + 1e-10)
        return self.custom_smooth_transition(mse_db)

    def custom_smooth_transition(self, mse_db, lower_bound=-50, upper_bound=50, smoothness=0.1):
        scale = smoothness
        mid_point = (upper_bound + lower_bound) / 2
        transition = lower_bound + (upper_bound - lower_bound) / (1 + torch.exp(-scale * (mse_db - mid_point)))
        transition = torch.clamp(transition, min=lower_bound, max=upper_bound)
        return transition.mean(), mse_db.mean()

    def ber_loss(self, model_output, targets, model_parameters=None, hparam=None):
        if isinstance(model_output, dict):
            # Use Mean Squared Error (MSE) loss for logits
            mse_loss = nn.functional.mse_loss(model_output['bits'], targets['bits'], reduction='mean')
        else:
            mse_loss = nn.functional.mse_loss(model_output, targets['bits'], reduction='mean')
        return mse_loss, calculate_ber(model_output, targets['bits'])
        
    def combined_ber_soi(self, model_output, targets, model_parameters=None):
        mse = nn.functional.mse_loss(model_output['soi'], targets['soi'], reduction='none')
        mse_db = 10 * torch.log10(mse + 1e-10)
        # Calculate the Mean Absolute Error (MAE) term
        mae = nn.functional.l1_loss(model_output['soi'], targets['soi'], reduction='mean')
        return mae, mse_db            
    

        
        
    def forward(self, model_output, targets, interference_sig_type, model_input=[], eps_sinr=1, Is_increment=False, model_parameters=None, DeepSup=None, hparam=None):
        
        if Is_increment:
            k = 10 ** (-eps_sinr / 10)
            reduced_interference = k * (model_input - targets['soi'])
            targets['soi'] = targets['soi'] + reduced_interference

        self.L1 = 0
        self.L2 = 0
        
        # L1 Regularization
        if self.l1_weight > 0:
            l1_reg = torch.tensor(0.).to(model_output['soi'].device)
            for param in model_parameters.parameters():
                l1_reg += torch.norm(param, 1)
            self.L1 = self.l1_weight * l1_reg
        
        # L2 Regularization
        if self.l2_weight > 0:
            l2_reg = torch.tensor(0.).to(model_output['soi'].device)
            for param in model_parameters.parameters():
                l2_reg += torch.norm(param, 2)
            self.L2 = self.l2_weight * l2_reg

        LOSS0, MSE = self.loss_fn(model_output, targets, model_parameters, hparam=hparam)
        
        if DeepSup is not None:
            # trans_de1, trans_de2 = DeepSup
            trans_de1 = DeepSup[0]

            target_for_de1, target_for_de2={}, {}
            
            target_for_de1['soi'] = downsample(targets['soi'], factor=2)
            # target_for_de2['soi'] = downsample(targets['soi'], factor=1)
    
            # Compute additional loss terms
            loss_de1, mse_de1 = self.loss_fn(trans_de1, target_for_de1)
            # loss_de2 = self.loss_fn(trans_de2, target_for_de2)
            loss_de2 = 0
    
            # Given a1 and a2
            # a1,a2=0.5,0
            a1 = 0.5
            # total_weight = 1 + a1 + a2
            total_weight = 1 + a1
            weight_loss0 = 1 / total_weight
            weight_loss_de1 = a1 / total_weight
            # weight_loss_de2 = a2 / total_weight
            
            # Compute weighted loss
            # weighted_loss = weight_loss0 * LOSS0 + weight_loss_de1 * loss_de1 + weight_loss_de2 * loss_de2
            
            t1 = weight_loss0 * LOSS0
            t2 = weight_loss_de1 * loss_de1
            weighted_loss = weight_loss0 * LOSS0 + weight_loss_de1 * loss_de1

            LOSS= weighted_loss + self.L1+self.L2
        else:
            LOSS= LOSS0 + self.L1+self.L2
            
        return LOSS, MSE


def downsample(target, factor):
    # Implement your downsampling method here
    # Example: Using average pooling for downsampling
    pool = nn.AvgPool1d(kernel_size=factor, stride=factor, padding=0)
    downsampled_target = pool(target)
    return downsampled_target

def calculate_ber(predictions, targets):
    predictions = torch.round(predictions)  # Round to nearest integer (0 or 1)
    errors = (predictions != targets).float().sum()
    ber = errors / targets.numel()  # Number of bit errors divided by total number of bits
    return ber.item()














class MyCustomLoss(nn.Module):
    def __init__(self, loss_type='mse'):
        super(MyCustomLoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, outputs, targets, actual_sinr, ber_score_type):
        return self.compute_loss(outputs, targets, actual_sinr, self.loss_type, ber_score_type)

    def compute_loss(self, outputs, targets, actual_sinr, loss_type='mse', ber_score_type='soft'):
        # Compute the MSE between the signals of interest
        mse = F.mse_loss(outputs['soi'], targets['soi']) 
        mse_std = mse + ( mse * torch.std(outputs['soi'] - targets['soi']) )
        
        # Calculate MSE in dB without any truncation
        mse_in_db_untruncated = 10 * torch.log10(torch.clamp(mse, min=1e-9))
        
        # MSE score with a floor at -50dB
        mse_score = 11*torch.clamp(mse_in_db_untruncated, min=-50) 
    
        # Compute the bit errors and BER per batch sample
        if outputs['bits'].float().shape[-1]==targets['bits'].float().shape[-1]:
            bit_errors = torch.sum(torch.abs(outputs['bits'].float() - targets['bits'].float()), dim=1) 
            total_bits = outputs['bits'].size(1)
            ber = bit_errors.float() / total_bits  
        else:
            bit_errors = torch.sum(torch.abs(targets['bits'].float() - targets['bits'].float()), dim=1) 
            total_bits = outputs['bits'].size(1)
            ber = bit_errors.float() / total_bits  

        
        # Compute the BER score
        if ber_score_type == 'hard':
            # Using the device of 'ber' tensor for new tensors
            device = ber.device
            ber_score = (torch.where(ber >= 1e-2, torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)) * actual_sinr).mean()
        elif ber_score_type == 'soft':
            # Apply the sigmoid function directly on the BER tensor
            ber_score = 11 * (((1 + (5 * torch.exp(-10 * (ber - 1e-4)))) / 6) * actual_sinr).mean()
        else:
            raise ValueError("Invalid ber_score_type. Use 'hard' or 'soft'.")
    
        # Determine the first loss based on the specified loss_type
        if loss_type == 'mse':
            first_loss = mse
        elif loss_type == 'mse_in_db':
            first_loss = mse_in_db_untruncated  # mean over the batches
        else:
            raise ValueError("Invalid loss_type. Use 'mse' or 'mse_in_db'.")
    
        # The total score combines the truncated MSE score and the BER score and is averaged across the batch
        total_score = (mse_score + ber_score)  # average of mse_score and ber_score for each batch sample
        # Return the specified losses and scores
        return first_loss, total_score, mse_in_db_untruncated , mse_score, ber_score, ber.mean(), mse_std

    @staticmethod
    def single_loss_calculator(output, target, loss_type):
        # Compute MSE
        mse = torch.nn.functional.mse_loss(output, target)
        
        # Compute MSE in dB
        mse_db = 10 * torch.log10(mse)
        
        # Compute truncated MSE in dB
        mse_score = 11*torch.clamp(mse_db, min=-50)
        
        # Compute squared error and its standard deviation in dB
        squared_error = (output - target) ** 2
        std_db = 10 * torch.log10(torch.std(squared_error))
        
        # Compute truncated standard deviation in dB
        std_score = 11*torch.clamp(std_db, min=-50)
        
        # Compute loss based on the given loss type
        if loss_type == 'mse_in_db':
            loss = mse_db
        elif loss_type == 'normalized_mse_in_db':
            norm_factor = torch.nn.functional.mse_loss(target, torch.zeros_like(target))
            normalized_mse = mse / norm_factor
            loss = 10 * torch.log10(normalized_mse)
        elif loss_type == 'mse':
            loss = mse
        elif loss_type == 'mae':  # Mean Absolute Error
            loss = torch.nn.functional.l1_loss(output, target)
        elif loss_type == 'bce':  # Binary Cross Entropy
            loss = torch.nn.functional.binary_cross_entropy(output, target.unsqueeze(1))
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Return the computed values
        return loss, mse_db, mse_score, std_score
