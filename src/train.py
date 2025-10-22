#============================================================================
 #3. train.py - Training Script for GPU
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

class DiseaseModelTrainer:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # For early stopping
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
    
    def train(self, train_loader, val_loader, epochs=100, lr=1e-3, 
              early_stopping_patience=5, use_adamw=True):
        """Train the model on GPU with efficient settings and early stopping"""
        
        # Use label smoothing for better generalization
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # AdamW optimizer with weight decay
        if use_adamw:
            optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
            print("Using AdamW optimizer")
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
            print("Using Adam optimizer")
        
        # Cosine annealing scheduler for better convergence
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        
        # Enable gradient clipping for stability
        max_grad_norm = 1.0
        
        # Enable mixed precision training for faster GPU computation
        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        use_amp = scaler is not None
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Batch size: {train_loader.batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Epochs: {epochs}")
        print(f"Early stopping patience: {early_stopping_patience}")
        print(f"Mixed precision training: {use_amp}")
        print(f"Weight decay: 1e-5")
        print("-" * 60)
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                
                # Mixed precision training
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100 * train_correct / train_total
            
            # Validation phase
            val_loss, val_acc = self.evaluate(val_loader, criterion, use_amp)
            
            # Learning rate scheduling
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Record metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Early stopping logic
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                }, 'models/best_model.pth')
                
                best_marker = " ✓ (Best)"
            else:
                self.patience_counter += 1
                best_marker = ""
            
            epoch_time = time.time() - start_time
            
            # Print progress
            print(f'Epoch [{epoch+1:3d}/{epochs}] | '
                  f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | '
                  f'LR: {current_lr:.6f} | Time: {epoch_time:.2f}s{best_marker}')
            
            # Early stopping check
            if self.patience_counter >= early_stopping_patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
                print(f"Best validation loss: {self.best_val_loss:.4f}")
                print(f"{'='*60}")
                break
        
        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nRestored best model from epoch {epoch - self.patience_counter + 1}")
        
        print(f'\nTraining completed!')
        print(f'Best Validation Loss: {self.best_val_loss:.4f}')
        print(f'Final Validation Accuracy: {self.val_accs[epoch - self.patience_counter]:.2f}%')
        
        return self.best_val_loss
    
    def evaluate(self, data_loader, criterion, use_amp=False):
        """Evaluate the model with optional mixed precision"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(data_loader)
        val_acc = 100 * val_correct / val_total
        
        return val_loss, val_acc
    
    def evaluate(self, data_loader, criterion, use_amp=False):
        """Evaluate the model with optional mixed precision"""
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in data_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                else:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_loss /= len(data_loader)
        val_acc = 100 * val_correct / val_total
        
        return val_loss, val_acc
    
    def plot_training_history(self):
        """Plot training history with early stopping marker"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs_trained = len(self.train_losses)
        epochs = range(1, epochs_trained + 1)
        
        # Loss plot
        ax1.plot(epochs, self.train_losses, label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, label='Val Loss', linewidth=2)
        
        # Mark best epoch
        best_epoch = self.val_losses.index(min(self.val_losses)) + 1
        ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax1.scatter([best_epoch], [self.val_losses[best_epoch-1]], color='red', s=100, zorder=5)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.train_accs, label='Train Accuracy', linewidth=2)
        ax2.plot(epochs, self.val_accs, label='Val Accuracy', linewidth=2)
        ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label=f'Best Epoch ({best_epoch})')
        ax2.scatter([best_epoch], [self.val_accs[best_epoch-1]], color='red', s=100, zorder=5)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        print("Training history plot saved to 'models/training_history.png'")
        plt.show()