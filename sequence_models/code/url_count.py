last_url = {}
user_seq = {}
pair_cnt = {}
max_val = 0
max_pair = None
with open('./text.txt') as f:
    line = f.readline()
    while True:
        line = f.readline().strip('\n')
        if line == '':
            break
        row = line.split('\t')
        user, url = row[1], row[2]
        if user not in last_url:
            last_url[user] = url
            user_seq[user] = set()
        else:
            pair = (last_url[user], url)
            # 只统计对应user下第一次出现的pair
            if pair not in user_seq[user]:
                user_seq[user].add(pair)

                if pair not in pair_cnt:
                    pair_cnt[pair] = 1
                else:
                    pair_cnt[pair] += 1

                if pair_cnt[pair] > max_val:
                    max_val = pair_cnt[pair]
                    max_pair = pair
        last_url[user] = url
print(user_seq)
print(pair_cnt)
print(f"max same pair: {max_pair}, max number: {max_val}")

