import re
import math
import random
import numpy as np
from extract_features import get_features_from_game_both, get_features_from_game_single
from plot_data import plot_no_moves, plot_rating_distribution

PGN_PATH = 'data/data.pgn'
STOCKFISH_PATH = 'data/stockfish.csv'

def read_games():
    games = []

    with open(PGN_PATH, 'r') as file:
        pgn_games = file.read()

        index = 1
        last_index = pgn_games.find('[Event', index)
        exit = False
        while last_index != -1:
            current_game = pgn_games[index-1:last_index]

            lines = current_game.split('\n')
            result_= lines[6]
            
            result = 2 #draw
            if '0-1' in result_:
                result = 0
            elif '1-0' in result_:
                result = 1

            white_elo = re.findall(r'\d+', lines[7])
            black_elo = re.findall(r'\d+', lines[8])

            if not white_elo or not black_elo: #Test data not providing the elo
                moves = ''.join(lines[8:])
                games.append(['Test', result, moves])
            else:
                moves = ''.join(lines[10:])
                white_elo = int(white_elo[0])
                black_elo = int(black_elo[0])

                games.append([result, white_elo, black_elo, moves])

            index = last_index + 1
            last_index = pgn_games.find('[Event', index)
            if last_index == -1 and not exit:
                last_index = len(pgn_games) - 1
                exit = True
    
    return games

def read_stockfish():
    with open(STOCKFISH_PATH, 'r') as file:
        stockfish_evals = file.readlines()

        games_eval = []
        no = 0
        for eval_row_ in stockfish_evals[1:]:
            actual_row = eval_row_.split(',')[1]

            evals_ = actual_row.split(' ')
            no+=1
            if(len(evals_) == 2):
                games_eval.append([])
            else:
                evals = []
                for i in range(0, len(evals_)):
                    if evals_[i] == 'NA' or evals_[i] == 'NA\n':
                        evals_[i] = evals_[i-1]
                    
                    
                    evals.append(int(evals_[i]))
            
                games_eval.append(evals)

    return games_eval


def create_dataset():
    games = read_games()
    stockfish = read_stockfish()
    
    assert(len(games) == len(stockfish))

    trainval_games = []
    test_games = []

    for i in range(0, len(games)):
        game, eval = games[i], stockfish[i]
        
        if game[0] == 'Test':
            test_games.append([game[1:], eval])
        else:
            game.append(eval)
            trainval_games.append([game])
    x = []
    y = []
    
    for game in trainval_games:
        game = game[0]
        result = game[0]
        white = game[1]
        black = game[2]
        cp_loss = game[-1]

        features = get_features_from_game_both(result, cp_loss)
        
        if features != 0:
            x.append(features)
            y.append([white, black])
    plot_rating_distribution(x, y)
    plot_no_moves(x)
    
    return x,y

def create_multi_shot_dataset():
    games = read_games()
    stockfish = read_stockfish()
    
    assert(len(games) == len(stockfish))

    trainval_games = []
    test_games = []

    for i in range(0, len(games)):
        game, eval = games[i], stockfish[i]
        
        if game[0] == 'Test':
            test_games.append([game[1:], eval])
        else:
            game.append(eval)
            trainval_games.append([game])

    first_x = []
    first_y = []
    
    second_x = []
    second_y = []

    for game in trainval_games:
        game = game[0]
        result = game[0]
        white = game[1]
        black = game[2]
        cp_loss = game[-1]

        avg_rating = (white+black)//2

        first_shot_features = get_features_from_game_single(result, cp_loss)

        second_shot_features = get_features_from_game_both(result, cp_loss)
        second_shot_features.append(avg_rating + random.randint(5,50))

        if not(second_shot_features == 0 or first_shot_features == 0):
            first_x.append(first_shot_features)
            first_y.append(avg_rating)

            second_x.append(second_shot_features)
            second_y.append([white, black])
            

    train_ratio = 0.85
    size = len(first_x)
    train_size = math.floor(size * train_ratio)
    

    first_train_x, first_train_y= first_x[0:train_size], first_y[0:train_size]
    second_train_x, second_train_y = second_x[0:train_size], second_y[0:train_size]

    first_val_x, first_val_y= first_x[train_size:size], first_y[train_size:size]
    second_val_x, second_val_y = second_x[train_size:size], second_y[train_size:size]

    return first_train_x, first_train_y, second_train_x, second_train_y, first_val_x, first_val_y, second_val_x, second_val_y

def read_from_lichess(str_analysis):
    move_analysis = str_analysis.split("[%eval")
    evals = []
    for _ in move_analysis[1:]:
        str_eval_ = _.split(' ')[1]
        str_eval = str_eval_[:-1]
        eval = math.floor(float(str_eval) * 100)
        evals.append(eval)
    
    return evals

def read_test_data():
    with open('data/test.pgn', 'r') as file:
        lines = file.readlines()

        if lines[0] == '0-1':
            result = 0
        elif lines[0] == '1-0':
            result = 1
        else:
            result = 2

        cp_loss = read_from_lichess(lines[1])

        features = get_features_from_game_both(result, cp_loss)
    return features

read_from_lichess("1. d4 { [%eval 0.0] } 1... Nf6 { [%eval 0.23] } 2. c4 { [%eval 0.29] } 2... e6 { [%eval 0.23] } 3. Nc3 { [%eval 0.05] } 3... Bb4 { [%eval 0.25] } 4. g3 { [%eval 0.0] } { E20 Nimzo-Indian Defense: Romanishin Variation } 4... O-O { [%eval 0.0] } 5. Bg2 { [%eval 0.0] } 5... d5 { [%eval 0.0] } 6. a3 { [%eval -0.23] } 6... Bxc3+ { [%eval -0.12] } 7. bxc3 { [%eval 0.0] } 7... dxc4 { [%eval 0.0] } 8. Nf3 { [%eval -0.29] } 8... c5 { [%eval -0.12] } 9. O-O { [%eval 0.13] } 9... cxd4 { [%eval -0.03] } 10. Qxd4 { [%eval -0.05] } 10... Nc6 { [%eval -0.03] } 11. Qxc4 { [%eval 0.0] } 11... e5 { [%eval -0.13] } 12. Bg5 { [%eval 0.0] } 12... h6 { [%eval 0.04] } 13. Rfd1 { [%eval -0.28] } 13... Be6 { [%eval -0.48] } 14. Rxd8 { [%eval -0.37] } 14... Bxc4 { [%eval -0.37] } 15. Rxa8 { [%eval -0.23] } 15... Rxa8 { [%eval -0.37] } 16. Bxf6 { [%eval -0.52] } 16... gxf6 { [%eval -0.18] } 17. Kf1 { [%eval -0.52] } 17... Rd8 { [%eval -0.47] } 18. Ke1 { [%eval -0.45] } 18... Na5 { [%eval -0.28] } 19. Rd1 { [%eval -0.39] } 19... Rc8 { [%eval -0.54] } 20. Nd2 { [%eval -0.2] } 20... Be6 { [%eval -0.28] } 21. c4 { [%eval -0.57] } 21... Bxc4 { [%eval -0.25] } 22. Nxc4 { [%eval -0.39] } 22... Rxc4 { [%eval -0.36] } 23. Rd8+ { [%eval -0.43] } 23... Kg7 { [%eval -0.42] } 24. Bd5 { [%eval -0.34] } 24... Rc7 { [%eval -0.47] } 25. Ra8 { [%eval -0.61] } 25... a6 { [%eval -0.56] } 26. Rb8 { [%eval -0.39] } 26... f5 { [%eval -0.63] } 27. Re8 { [%eval -0.62] } 27... e4 { [%eval -0.63] } 28. g4?! { (-0.63 → -1.31) Inaccuracy. Rd8 was best. } { [%eval -1.31] } (28. Rd8 Kf6 29. h4 Ke5 30. f4+ exf3 31. exf3 f4 32. g4 b5 33. Be4 Nc4 34. Rd5+ Kf6) 28... Rc5 { [%eval -1.22] } 29. Ba2 { [%eval -1.77] } 29... Nc4?! { (-1.77 → -0.74) Inaccuracy. fxg4 was best. } { [%eval -0.74] } (29... fxg4) 30. a4?! { (-0.74 → -1.80) Inaccuracy. Bxc4 was best. } { [%eval -1.8] } (30. Bxc4 Rxc4) 30... Nd6 { [%eval -1.56] } 31. Re7 { [%eval -3.25] } 31... fxg4? { (-3.25 → -1.84) Mistake. Rc2 was best. } { [%eval -1.84] } (31... Rc2) 32. Rd7 { [%eval -1.94] } 32... e3 { [%eval -1.83] } 33. fxe3 { [%eval -1.86] } 33... Ne4 { [%eval -1.84] } 34. Kf1 { [%eval -1.75] } 34... Rc1+ { [%eval -1.07] } 35. Kg2 { [%eval -1.03] } 35... Rc2 { [%eval -1.15] } 36. Bxf7 { [%eval -1.28] } 36... Rxe2+ { [%eval -1.02] } 37. Kg1 { [%eval -1.1] } 37... Re1+ { [%eval -1.06] } 38. Kg2 { [%eval -0.88] } 38... Re2+ { [%eval -1.24] } 39. Kg1 { [%eval -1.4] } 39... Kf6 { [%eval -1.38] } 40. Bd5 { [%eval -1.46] } 40... Rd2 { [%eval -1.34] } 41. Rf7+ { [%eval -1.46] } 41... Kg6 { [%eval -1.28] } 42. Rd7?? { (-1.28 → -3.55) Blunder. Re7 was best. } { [%eval -3.55] } (42. Re7 Rxd5 43. Rxe4 Kg5 44. h3 gxh3 45. Kh2 Kf6 46. Rh4 h5 47. Rb4 b5 48. e4 Rg5) 42... Ng5 { [%eval -4.13] } 43. Bf7+ { [%eval -4.21] } 43... Kf5? { (-4.21 → -2.46) Mistake. Kf6 was best. } { [%eval -2.46] } (43... Kf6 44. Rxd2 Nf3+ 45. Kg2 Nxd2 46. Bd5 b5 47. axb5 axb5 48. Kg3 Kg5 49. h3 gxh3 50. Kxh3) 44. Rxd2 { [%eval -2.4] } 44... Nf3+ { [%eval -2.79] } 45. Kg2 { [%eval -3.45] } 45... Nxd2 { [%eval -2.55] } 46. a5 { [%eval -2.88] } 46... Ke5 { [%eval -3.1] } 47. Kg3 { [%eval -3.82] } 47... Nf1+ { [%eval -3.9] } 48. Kf2?! { (-3.90 → -5.97) Inaccuracy. Kxg4 was best. } { [%eval -5.97] } (48. Kxg4 Nxh2+) 48... Nxh2 { [%eval -6.44] } 49. e4 { [%eval -6.17] } 49... Kxe4 { [%eval -7.81] } 50. Be6 { [%eval -9.07] } 50... Kf4 { [%eval -9.89] } 51. Bc8 { [%eval -10.4] } 51... Nf3 { [%eval -10.03] } 52. Bxb7 { [%eval -10.85] } 52... Ne5 { [%eval -10.99] } 53. Bxa6 { [%eval -11.53] } 53... Nc6 { [%eval -12.0] } 54. Bb7 { [%eval -11.85] } 54... Nxa5 { [%eval -11.89] } 55. Bd5 { [%eval -12.53] } 55... h5 { [%eval -12.33] } 56. Bf7 { [%eval -14.51] } 56... h4 { [%eval -12.88] } 57. Bd5 { [%eval -13.5] } { Black wins. } 0-1")

