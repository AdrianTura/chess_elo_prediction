blunder_threshold = 500
mistake_threshold = 300
innacuracy_threshold = 100
strong_threshold = 60
best_threshold = 30

no_opening_moves_threshold = 21
no_middle_moves_threshold = 51


class FeatureExtractor():
    no_blunders = 0
    no_mistakes = 0
    no_innacuracy = 0
    no_strong = 0
    no_best = 0
    avg_loss = 0
    avg_opening_loss = 0
    avg_middle_loss = 0
    avg_endgame_loss = 0

    def __init__(self, color):
        self.color = color

    def register_move(self, current_eval, last_eval, move_no):
        def calculate_avg(current_avg, current_value, no_moves):
            return (current_avg * (no_moves-1) + current_value)/no_moves
        
        error = last_eval - current_eval

        if error > blunder_threshold:
            self.no_blunders += 1
        elif error > mistake_threshold:
            self.no_mistakes += 1
        elif error > innacuracy_threshold:
            self.no_innacuracy += 1
        elif error < best_threshold:
            self.no_best += 1
        elif error < strong_threshold:
            self.no_strong += 1

        if move_no < no_opening_moves_threshold:
            self.avg_opening_loss = calculate_avg(self.avg_opening_loss, error, move_no+1)
        elif move_no < no_middle_moves_threshold:
            self.avg_middle_loss = calculate_avg(self.avg_middle_loss, error, move_no-no_opening_moves_threshold+1)
        else:
            self.avg_endgame_loss = calculate_avg(self.avg_endgame_loss, error, move_no-no_middle_moves_threshold+1)

        self.avg_loss = calculate_avg(self.avg_loss, error, move_no+1)

    def get_features(self, no_moves):
        return [self.no_blunders/no_moves, self.no_mistakes/no_moves, self.no_innacuracy/no_moves, self.no_strong/no_moves, self.no_best/no_moves, abs(self.avg_loss)/no_moves, \
                abs(self.avg_opening_loss), abs(self.avg_middle_loss), abs(self.avg_endgame_loss)]

def get_features_from_game_both(result, cp_list):
    white = FeatureExtractor('white')
    black = FeatureExtractor('black')
    no_moves = 0
    no_swings = 0

    if(cp_list == 0 or cp_list == []):
        return 0
    
    for i in range(1, len(cp_list)):
        if cp_list[i] * cp_list[i-1] < 0:
                no_swings += 1
        if i%2 == 1:
            black.register_move(cp_list[i], cp_list[i-1], no_moves)
        else:
            no_moves += 1
            white.register_move(cp_list[i], cp_list[i-1], no_moves)
    
    return [no_moves, result] + white.get_features(no_moves) + black.get_features(no_moves)

def get_features_from_game_single(result, cp_list):
    games = FeatureExtractor('both')
    no_moves = 0
    
    if(len(cp_list) == 0):
        return 0
    
    for i in range(1, len(cp_list)):
        games.register_move(cp_list[i], cp_list[i-1], no_moves)
        if i%2 == 0:
            no_moves += 1

    return [no_moves, result] + games.get_features(no_moves)

def main():
    features = get_features_from_game_both(1,[4,2, 21, 5, 53 ,35, 45, 37, 54, 10, 22, 8 ,48, 30, 17, 13, 35, -12, 31, \
                                              13, 31, -22, 20, -18, -30, -27, 6, -11, 74, 53, 50, 56, 55, 68, 58, 44, 59, \
                                                -10, 12, -11, 0, -11, 29, 27, 31, 26, 24, 28, 42, 36, 38, 46, 36, 44, 45, 40, \
                                                    36 ,34 ,68, 55, 53, 26, 89 ,88 ,79 ,90 ,103, 72, 117, 133, 135, 134, 163, \
                                                        169, 167, 198, 182, 196])

    print(features)
