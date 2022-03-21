



def one_hot(p: int, ncodes: int):
    if p < 0:
        raise ValueError('p should be positive')
    p_str = format(1 << p, f'0{ncodes+2}b')[2:]
    return list(map(int, list(p_str)))