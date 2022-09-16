import argparse
import os

from utils import Part, Comparator

def run():
    
    # load all signatures for all instances of a part
    conlid = Part("CONLID")  # container and lid together
    con = Part("CON")  # container alone
    lid = Part("LID")  # lid alone
    sen = Part("SEN")  # sensor alone

    # display the instance names for a part
    sen.list_instances()

    # load an instance object by instance ID
    sen_x1 = sen.get_instance("x1")
    sen_x2 = sen.get_instance("x2")
    print(type(sen_x1))

    # show all signature names for the selected instance
    sen_x1.list_signatures()
    
    sen_x1_sig1 = sen_x1.get_signature("1")
    sen_x1_sig2 = sen_x1.get_signature("2")
    sen_x2_sig1 = sen_x2.get_signature("1")
    sen_x1.get_average_signature()
    
    # overloaded print to get summary of the signature
    print(sen_x1_sig1)

    # generate synthetic signatures
    sen_x1.avg_signature.generate_synthetic(plot=False)
    noise_avg_real, noise_std_real, noise_avg_imag, noise_std_imag = sen_x1_sig1.generate_synthetic(plot=False)
    
    # compare noise quality to source signature
    print("--------------")
    print(f"avg of avg noise value vs original real: {noise_avg_real} vs {sen_x1_sig1.norm_avg}, diff = {abs(noise_avg_real - sen_x1_sig1.norm_avg)}")
    print(f"avg of std noise value vs original real: {noise_std_real} vs {sen_x1_sig1.norm_std}, diff = {abs(noise_std_real - sen_x1_sig1.norm_std)}")
    print(f"avg of avg noise value vs original imag: {noise_avg_imag} vs {sen_x1_sig1.norm_avg_imag}, diff = {abs(noise_avg_imag - sen_x1_sig1.norm_avg_imag)}")
    print(f"avg of std noise value vs original imag: {noise_std_imag} vs {sen_x1_sig1.norm_std_imag}, diff = {abs(noise_std_imag - sen_x1_sig1.norm_std_imag)}")
    print("--------------")

    # compare two different signatures of the same instance
    x1_sig1_vs_sig2 = Comparator(sen_x1_sig1, sen_x1_sig2)
    x1_sig1_vs_x2_sig1 = Comparator(sen_x1_sig1, sen_x2_sig1)

    # calculate the difference according to a few metrics
    x1_sig1_vs_sig2.compare()
    x1_sig1_vs_x2_sig1.compare()


if __name__ == "__main__":
    run()