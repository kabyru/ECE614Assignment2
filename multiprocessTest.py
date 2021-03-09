from multiprocessing import Pool

def sayHello(data):
    print(data)
    print("hello " + str(data['data_inputs']) + " " + str(data['data_inputs_2']))

if __name__ == '__main__':
    data_inputs = ["Kaleb", "Nathan", "Brian"]
    data_inputs_2 = ["Big", "Succ", "Here"]
    data = [{"data_inputs":data_inputs[0], "data_inputs_2":data_inputs_2[0]}]
    pool = Pool()
    pool.map(sayHello,data)