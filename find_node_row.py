SINK_NODE_1 = 1281123728
SINK_NODE_2 = 1281108573
SINK_NODE_3 = 1281090837
SINK_NODE_4 = 1281069755
SINK_NODE_5 = 1281069767
SINK_NODE_6 = 1281069792
SINK_NODE_7 = 1281070290
SINK_NODE_8 = 1281070506
SINK_NODE_9 = 1281071040
SINK_NODE_10 = 1281071912
SINK_NODE_11 = 1281072772
SINK_NODE_12 = 1281074111
SINK_NODE_13 = 1281074917
SINK_NODE_14 = 1281075355
SINK_NODE_15 = 1281076318
SINK_NODE_16 = 1281076637
SINK_NODE_17 = 1281079448
SINK_NODE_18 = 1281079623
SINK_NODE_19 = 1281083099
SINK_NODE_20 = 1281083165
SINK_NODE_21 = 1281084228
SINK_NODE_22 = 1281085256
SINK_NODE_23 = 1281086297
SINK_NODE_24 = 1281086588
SINK_NODE_25 = 1281088215
SINK_NODE_26 = 1281088329
SINK_NODE_27 = 1281088408
SINK_NODE_28 = 1281089492
SINK_NODE_29 = 1281089604
SINK_NODE_30 = 1281090120
SINK_NODE_31 = 1281091172
SINK_NODE_32 = 1281091320
SINK_NODE_33 = 1281093209
SINK_NODE_34 = 1281093795
SINK_NODE_35 = 1281099564
SINK_NODE_36 = 1281101893
SINK_NODE_37 = 1281102666
SINK_NODE_38 = 1281104414
SINK_NODE_39 = 1281106683
SINK_NODE_40 = 1281109926
SINK_NODE_41 = 1281123728
SINK_NODE_42 = 1281125857
SINK_NODE_43 = 1281128465
SINK_NODE_44 = 1281128574
SINK_NODE_45 = 1281129004
SINK_NODE_46 = 1281133963
SINK_NODE_47 = 1281135083
SINK_NODE_48 = 1281135306
SINK_NODE_49 = 1281136514
SINK_NODE_50 = 1281137429
SINK_NODE_51 = 1281137439
SINK_NODE_52 = 1281140647
SINK_NODE_53 = 1281141992
SINK_NODE_54 = 1281143603
SINK_NODE_55 = 1281145920
SINK_NODE_56 = 1281146897
SINK_NODE_57 = 1281146946
SINK_NODE_58 = 1281148021
SINK_NODE_59 = 1281148891
SINK_NODE_60 = 1281148941
SINK_NODE_61 = 3762907343
SINK_NODE_62 = 3762914192
SINK_NODE_63 = 3762984021
SINK_NODE_64 = 5106582130
SINK_NODE_65 = 10874967889
SINK_NODE_66 = 10874967922
SINK_NODE_67 = 11476114070
SINK_NODE_68 = 11484371804
SINK_NODE_69 = 11486371746
SINK_NODE_70 = 11486600663
SINK_NODE_71 = 11486765638
SINK_NODE_72 = 11486990996
SINK_NODE_73 = 11492114090
SINK_NODE_74 = 11493056647
SINK_NODE_75 = 11493782008
SINK_NODE_76 = 11496903660
SINK_NODE_77 = 11497142496
SINK_NODE_78 = 11506868426

# filepath: /home/algorithm/project/EvacuationPJ/find_node_row.py
import pandas as pd

def find_node_rows():
    """
    Find the row numbers for all sink nodes in the CSV file.
    Returns a dictionary mapping sink node variables to their row numbers.
    """
    # Read the CSV file
    df = pd.read_csv('suzu_nodes_detailed.csv')
    
    # Get all sink node values from globals
    sink_nodes = {}
    for var_name, var_value in globals().items():
        if var_name.startswith('SINK_NODE_'):
            sink_nodes[var_name] = var_value
    
    # Find row numbers for each sink node
    results = {}
    for var_name, node_id in sink_nodes.items():
        # Find the row where osmid matches the node_id
        row_index = df[df['osmid'] == node_id].index
        if len(row_index) > 0:
            # Add 1 because pandas index is 0-based but we want 1-based row numbers
            # Add 1 more to account for the header row
            row_number = row_index[0] + 2
            results[var_name] = row_number
            print(f"{var_name} = {row_number}")
        else:
            print(f"{var_name} = NOT FOUND")
    
    return results

if __name__ == "__main__":
    find_node_rows()


