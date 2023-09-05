import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
manager = plt.get_current_fig_manager()
manager.full_screen_toggle()

def print_figure(x, y, xtitle, ytitle, figtitle):
    plt.figure(figsize=(10,6))
    plt.bar(y, x)

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(figtitle)

    plt.savefig('plots/{}.png'.format(figtitle))
    plt.close()

def plot_rating_distribution(features, y):
    no_categories = 13
    counters = avg_counter = loss = loss_opening = loss_middle = loss_endgame = swings = \
        no_blunders = no_mistakes = no_innacuracies = no_strong = no_best = [0]*no_categories
    
    def rating_to_index(rating):
        index = rating//100 - 15
        if index < 0:
            index = 0
        elif index > no_categories-1:
            index = no_categories-1

        return index

    for i in range(0, len(features)):
        white, black = y[i][0], y[i][1]

        avg_rating = (white + black)//2

        f = features[i]
        w_blunders, w_mistakes, w_innacuracy, w_strong, w_best, w_loss, w_ope_loss, w_mid_loss, w_end_loss= f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9], f[10]
        b_blunders, b_mistakes, b_innacuracy, b_strong, b_best, b_loss, b_ope_loss, b_mid_loss, b_end_loss= f[11], f[12], f[13], f[14], f[15], f[16], f[17], f[18], f[19]
        #sw = f[20]

        avg_index = rating_to_index(avg_rating)
        #swings[avg_index] += sw
        avg_counter[avg_index] += 1

        w_index = rating_to_index(white)
        b_index = rating_to_index(black)

        loss[w_index] += w_loss
        no_blunders[w_index] += w_blunders
        no_mistakes[w_index] += w_mistakes
        no_innacuracies[w_index] += w_innacuracy
        no_strong[w_index] += w_strong
        no_best[w_index] += w_best
        loss_opening[w_index] += w_ope_loss
        if w_mid_loss != 0:
            loss_middle[w_index] += w_mid_loss
        if w_end_loss != 0:
            loss_endgame[w_index] += w_end_loss

        loss[b_index] += b_loss
        no_blunders[b_index] += b_blunders
        no_mistakes[b_index] += b_mistakes
        no_innacuracies[b_index] += b_innacuracy
        no_strong[b_index] += b_strong
        no_best[b_index] += b_best
        loss_opening[b_index] += b_ope_loss
        if b_mid_loss != 0:
            loss_middle[b_index] += b_mid_loss
        if b_end_loss != 0:
            loss_endgame[b_index] += b_end_loss

        counters[b_index] += 1
        counters[w_index] += 1
        
    def avg_(arr, ctr_arr):
        return [arr[i]/ctr_arr[i] for i in range(0, len(arr))]
    
    avg_loss = avg_(loss, counters)
    avg_blunders = avg_(no_blunders, counters)
    avg_mistakes = avg_(no_mistakes, counters)
    avg_innacuracies = avg_(no_innacuracies, counters)
    avg_strong = avg_(no_strong, counters)
    avg_best = avg_(no_best, counters)
    avg_openings = avg_(loss_opening, counters)
    avg_middle = avg_(loss_middle, counters)
    avg_endgame = avg_(loss_endgame, counters)

    #avg_swings = avg_(swings, avg_counter)
    columns = ['u1600', 'u1700', 'u1800', 'u1900', 'u2000', 'u2100', 'u2200', 'u2300', 'u2400', 'u2500', 'u2600', 'u2700', '>2700']
    print_figure(avg_counter, columns, 'Average rating', 'Number of games', 'rating_distribution')
    print_figure(avg_loss, columns, 'Rating', 'Average centipawn loss', 'centipawn_loss')
    print_figure(avg_blunders, columns, 'Rating', 'Number of blunders', 'blunders')
    print_figure(avg_mistakes, columns, 'Rating', 'Number of mistakes', 'mistakes')
    print_figure(avg_innacuracies, columns, 'Rating', 'Number of innacuracies', 'innacuracy')
    print_figure(avg_strong, columns, 'Rating', 'Number of strong moves', 'strong')
    print_figure(avg_best, columns, 'Rating', 'Number of strong moves', 'best')
    #print_figure(avg_swings, columns, 'Rating', 'Swings', 'swing')
    print_figure(avg_openings, columns, 'Rating', 'Opening loss', 'opening_loss')
    print_figure(avg_middle, columns, 'Rating', 'Middlegame loss', 'middle_loss')
    print_figure(avg_endgame, columns, 'Rating', 'Endgame loss', 'endgame_loss')
    plt.bar(columns, swings)

def plot_no_moves(features):
    no_categories = 9
    counters = [0]*no_categories

    for game_feature in features:
        no_moves = game_feature[0]

        index = no_moves//10

        if index >= no_categories:
            index = no_categories -1

        counters[index] += 1
    
    columns = ['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '>80']
    print_figure(counters, columns, 'Number of games', 'Number of moves', 'number_of_moves')
    

def plot_results(results):
    names = []
    mae_errors = []
    r2_errors = []

    for r in results:
        name, mae, mse, rmse, r2, adjusted_r2 = r[0], r[1], r[2], r[3], r[4], r[5]

        names.append(name)
        mae_errors.append(mae)
        r2_errors.append(r2)

    print_figure(mae_errors, names, 'model', 'value', 'mae')
    print_figure(r2_errors, names, 'model', 'value', 'r2')
