import numpy as np

class Easy21(object):
    # Implement easy21
    # state: (dealer's first card, player's hand)
    # action: 1(hit); 0(stick)
    # reward: -1(lose); 0(draw); 1(win)
    
    def __init__(self):
        self.new_game() 
        
    def new_game(self):
        self.dealer_first_card = self.dealer_hand = np.random.randint(1, 11)
        self.player_hand = np.random.randint(1, 11)
        self.game_end = False
        
    def deal_card(self):
        color = np.random.randint(3)
        value = np.random.randint(1, 11)
        if color <= 1:
            return value
        else:
            return -value
    
    def observe(self):
        # return current state
        return [self.dealer_first_card, self.player_hand]
    
    def is_terminal(self):
        return self.game_end
    
    def step(self, action):
        if self.is_terminal():
            self.new_game()
        if action == 1:
            # Player hits
            self.player_hand += self.deal_card()
            if 1 < self.player_hand < 22:
                # continue for another action, zero reward
                reward = 0
                self.game_end = False
            else: 
                # Player goes bust
                self.player_hand = 0
                reward = -1
                self.game_end = True
        else:
            # Player sticks
            while self.dealer_hand < 17:
                # Dealer hits
                self.dealer_hand += self.deal_card()
                if 1 < self.dealer_hand < 22:
                    continue
                else:
                    # Dealer goes bust
                    self.dealer_hand = 0
                    reward = 1
                    self.game_end = True
                    return ([self.dealer_first_card, self.player_hand], reward)
            # Dealer sticks
            if self.dealer_hand > self.player_hand:
                reward = -1
            elif self.dealer_hand == self.player_hand:
                reward = 0
            else:
                reward = 1
            self.game_end = True
        return ([self.dealer_first_card, self.player_hand], reward)
