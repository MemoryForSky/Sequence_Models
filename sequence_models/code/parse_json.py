def print_dict(k, v, res):
    if isinstance(v, dict):
        for kk in v.keys():
            print_dict(k + '_' + kk, v[kk], res)
    else:
        res[k] = v


# 1.test origin
input = {"userid": "121343", "message": {"id": "111", "content": {"text": "hello", "language": "english" },
                                         "create_time": "20170224 14:51:00"}}
output = {"userid": "121343", "message_id": "111", "message_content_text": "hello",
          "message_content_language": "english", "message_create_time": "20170224 14:51:00"}

result = {}
for k in input.keys():
    print_dict(k, input[k], result)
assert result == output, "Error"


# 2.test empty dict
input = {}
output = {}
result = {}
for k in input.keys():
    print_dict(k, input[k], result)
assert result == output, "Error"

# 3.test one value
input = {"userid": "121343"}
output = {"userid": "121343"}
result = {}
for k in input.keys():
    print_dict(k, input[k], result)
assert result == output, "Error"

# 4.test empty dict in json
input = {"userid": "121343", "message": {"id": "111", "content": {},
                                         "create_time": "20170224 14:51:00"}}
output = {"userid": "121343", "message_id": "111", "message_create_time": "20170224 14:51:00"}

result = {}
for k in input.keys():
    print_dict(k, input[k], result)
assert result == output, "Error"
