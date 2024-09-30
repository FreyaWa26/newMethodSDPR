import os
class Dat:
    def __init__(self):
        self.ind_idx = []
        self.n_ind = 0
        self.n_snp = 0
        self.pheno = None
        self.n_cov = 0
        self.covar = None
        self.chr = []
        self.pos = []
        self.id = []
        self.ref = []
        self.alt = []
        self.geno1 = None
        self.geno2 = None
        self.n_anc1 = None
        self.n_anc2 = None

def get_size_vcf(pheno_path, geno_path, dat):
    n_ind = 0
    n_invalid = 0
    n_snp = 0

    with open(pheno_path, 'r') as infile1:
        i = 0
        for line in infile1:
            parts = line.split()
            if len(parts) < 3:
                continue
            id1, id2, y = parts[0], parts[1], parts[2]
            try:
                float(y)
                dat.ind_idx.append(i)
                n_ind += 1
            except ValueError:
                n_invalid += 1
            i += 1
    dat.n_ind = n_ind
    print(f"Warning: {n_invalid} individuals with invalid phenotypes.")

    with open(geno_path, 'r') as infile2:
        for line in infile2:
            if line.startswith("##"):
                continue
            elif line.startswith("#"):
                continue
            else:
                n_snp += 1

    dat.n_snp = n_snp
    print(f"In total {n_snp} SNPs and {n_ind} individuals to be read.")

def read_pheno(pheno_path, dat):
    print(f"Reading phenotype file from: {pheno_path}.")

    pheno = [0.0] * dat.n_ind

    with open(pheno_path, 'r') as infile:
        i = 0
        idx = 0
        for line in infile:
            parts = line.split()
            if len(parts) < 3:
                continue
            id1, id2, y = parts[0], parts[1], parts[2]
            if i == dat.ind_idx[idx]:
                pheno[idx] = float(y)
                idx += 1
            i += 1
            if idx >= dat.n_ind:
                break

    dat.pheno = pheno

    print(f"Read phenotype from {idx} individuals.")

def read_cov(cov_path, dat):
    print(f"Reading covariate file from: {cov_path}.")

    with open(cov_path, 'r') as infile:
        lines = infile.readlines()
    
    if not lines:
        return
    
    # Determine the number of covariates
    header = lines[0].strip().split('\t')
    n_cov = len(header)
    print(f"Reading {n_cov} covariates.")
    dat.n_cov = n_cov

    # Initialize the covariate matrix
    dat.covar = [[0.0] * dat.n_ind for _ in range(n_cov)]

    idx = 0
    for i, line in enumerate(lines):
        if idx >= dat.n_ind:
            print('break')
            break  # Ensure no out-of-bounds errors
        if i == dat.ind_idx[idx]:
            tokens = line.strip().split('\t')
            for n_cov_idx in range(n_cov):
                dat.covar[n_cov_idx][idx] = float(tokens[n_cov_idx])
            idx += 1
#e.g
#dat = Dat()
#get_size_vcf('/gpfs/gibbs/pi/zhao/gz222/SDPR_admix/Real/phenotype/SDPR/height.txt', '/gpfs/gibbs/pi/zhao/gz222/SDPR_admix/Real/genotype/Topmed/22.filter.vcf', dat)
#read_pheno('/gpfs/gibbs/pi/zhao/gz222/SDPR_admix/Real/phenotype/SDPR/height.txt', dat)
#read_cov('/gpfs/gibbs/pi/zhao/gz222/SDPR_admix/Real/phenotype/covar.txt', dat)

# Example usage:
# dat = Dat()
# # Assuming get_size_vcf and read_pheno have been called before and dat is populated
# get_size_vcf('pheno_path.txt', 'geno_path.vcf', dat)
# read_pheno('pheno_path.txt', dat)
# read_cov('cov_path.txt', dat)
# read_lanc('geno_path.vcf', 'msp_path.txt', dat)

def read_lanc(vcf_path, msp_path, dat):
    print(f"Reading RFmix msp file from: {msp_path}.")
    # Skip lines starting with #
    msp_lines = [];i =0
    with open(msp_path, 'r') as mspfile:
        for line in mspfile:
            i = i+1
            if i>2:
                msp_lines.append(line)

    print(f"Reading VCF file from: {vcf_path}.")
    infile = open(vcf_path, 'r')
    # Skip the header of the VCF file
    for line in infile:
        if line.startswith('##'):
            continue
        elif line.startswith('#'):
            n_ind = len(line.strip().split('\t')) - 9
            break
    
                
    # Read the pos from the first line of the VCF file after the header
    line2 = infile.readline().strip()
    tokens = line2.split('\t')
    chr_vcf = int(tokens[0])
    pos = int(tokens[1])


    idx_snp = 0
    dat.geno1 = [[0.0] * dat.n_ind for _ in range(dat.n_snp)]
    dat.geno2 = [[0.0] * dat.n_ind for _ in range(dat.n_snp)]
    dat.n_anc1 = [0.0] * dat.n_snp
    dat.n_anc2 = [0.0] * dat.n_snp

    for line1 in msp_lines:
        tokens1 = line1.strip().split('\t')
        chr_msp = int(tokens1[0])
        spos = int(tokens1[1])
        epos = int(tokens1[2])
        hap_lanc = [int(tok) for tok in tokens1[6:6 + 2 * n_ind]]
        
        if any(h != 0 and h != 1 for h in hap_lanc):
            print("RFmix field must be either 0 or 1.")
            return

        if chr_vcf != chr_msp or pos != spos:
            print(f"Inconsistent starting position: chr_vcf: {chr_vcf} chr_msp: {chr_msp} pos: {pos} spos: {spos}")
            return

        while (chr_vcf == chr_msp and spos <= pos < epos) or idx_snp == dat.n_snp - 1:
            if idx_snp == dat.n_snp - 1:
                assert chr_vcf == chr_msp and pos == epos

            tokens2 = line2.split('\t')
            dat.chr.append(tokens2[0])
            dat.pos.append(tokens2[1])
            dat.id.append(tokens2[2])
            dat.ref.append(tokens2[3])
            dat.alt.append(tokens2[4])

            k = 0
            for idx in range(9, len(tokens2)):
                if idx - 9 not in dat.ind_idx:
                    continue
                parts = tokens2[idx].split(':')
                genotype = parts[0].split('|')
                
                #genotype = tokens2[idx].split('|');print(genotype[:10])
                if '.' in genotype:
                    print("Missing genotype not supported yet.")
                    return

                genotype = [int(allele) for allele in genotype]

                if hap_lanc[2 * (idx - 9)] == 0:
                    dat.geno1[idx_snp][k] += genotype[0]
                    dat.n_anc1[idx_snp] += 1
                else:
                    dat.geno2[idx_snp][k] += genotype[0]
                    dat.n_anc2[idx_snp] += 1

                if hap_lanc[2 * (idx - 9) + 1] == 0:
                    dat.geno1[idx_snp][k] += genotype[1]
                    dat.n_anc1[idx_snp] += 1
                else:
                    dat.geno2[idx_snp][k] += genotype[1]
                    dat.n_anc2[idx_snp] += 1

                k += 1

            idx_snp += 1
            line2 = infile.readline().strip()
            tokens2 = line2.split('\t')
            if tokens2[0]:
                chr_vcf = int(tokens2[0])
            else:
                print("tokens2[0] is empty")
                break
            pos = int(tokens2[1])
        print(f" Read {idx_snp+1}SNPs")
    print(f"Read {idx_snp + 1} SNPs from {dat.n_ind} individuals.")
    infile.close()


#read_lanc('/gpfs/gibbs/pi/zhao/gz222/SDPR_admix/Real/genotype/Topmed/22.filter.vcf', '/gpfs/gibbs/pi/zhao/gz222/SDPR_admix/Real/genotype/Topmed/Rfmix/chr_22.msp.tsv', dat)
