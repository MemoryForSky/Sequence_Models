input = [['u1', 1, 100],
         ['u2', 1, 200],
         ['u3', 2, 500],
         ['u1', 3, 200],
         ['u3', 7, 200],
         ['u1', 8, 300],
         ['u1', 20, 400],
         ['u3', 20, 50],
         ['u2', 20, 300],
         ['u1', 22, 500],
         ['u2', 30, 600],
         ['u3', 30, 700]]

tm_delta_target = 20
amt_target = 800
user_last = {}
user_amt = {}
user_match = []
for index in range(len(input)):
    user = input[index][0]
    tm = input[index][1]
    amt = input[index][2]

    if user in user_match:
        continue

    if user not in user_last:
        user_last[user] = index
        user_amt[user] = amt

    else:
        last_index = user_last[user]
        if tm - input[last_index][1] > tm_delta_target:
            if user_amt[user] > amt_target:
                user_match.append(user)
            else:
                user_amt[user] += amt
                for i in range(last_index + 1, index):
                    if input[i][0] == user:
                        last_index = user_last[user]
                        user_last[user] = i
                        user_amt[user] -= input[last_index][2]
                        if tm - input[i][1] <= tm_delta_target:
                            if user_amt[user] > amt_target:
                                user_match.append(user)
                                break
        else:
            user_amt[user] += amt
print(user_amt)
print(user_match)



