# optimized to work at 2400x mag

def beamshift_test():
    import DigitalMicrograph as DM
    DM.ClearResults()

    import temscript
    import time
    test = temscript.RemoteMicroscope(('192.168.21.1',8080))
    test2 = temscript.RemoteMicroscope(('192.168.21.1',8081))
    family = test.get_family()
    print(family)
    print(type(family))
    test.set_beam_shift((0,0))
    a = test.get_beam_shift()
    print(a)
    shift = 0.00002
    sleepper = 0.1

    for _ in range(2):
        test.set_beam_shift((a[0], a[1]))
        time.sleep(sleepper)
        test.set_beam_shift((a[0]+shift, a[1]))
        time.sleep(sleepper)
        test.set_beam_shift((a[0], a[1]+shift))
        time.sleep(sleepper)
        test.set_beam_shift((a[0]-shift, a[1]))
        time.sleep(sleepper)
        test.set_beam_shift((a[0], a[1]-shift))
        time.sleep(sleepper)
        test.set_beam_shift((a[0], a[1]))
    print('done function')
    return