use bit_vec::BitVec;
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use rand::distributions::{Distribution, Standard};
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromPrimitive)]
pub enum NeuronConnectionType {
    InternalInput = 0,
    InternalOutput,
    InternalInternal,
    InputOutput,
}

impl Distribution<NeuronConnectionType> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> NeuronConnectionType {
        FromPrimitive::from_u8(rng.gen_range(0..4)).unwrap()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Gene {
    weight: u16,
    input_number: u8,
    output_number: u8,
    connection: NeuronConnectionType,
}

impl Distribution<Gene> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Gene {
        Gene::new(rng.gen(), rng.gen(), rng.gen(), rng.gen())
    }
}

impl Gene {
    fn new(
        weight: u16,
        input_number: u8,
        output_number: u8,
        connection: NeuronConnectionType,
    ) -> Gene {
        Gene {
            weight,
            input_number,
            output_number,
            connection,
        }
    }

    pub fn random() -> Gene {
        rand::thread_rng().gen()
    }

    pub fn weight(&self) -> u16 {
        self.weight
    }

    pub fn input_number(&self) -> u8 {
        self.input_number
    }

    pub fn output_number(&self) -> u8 {
        self.output_number
    }

    pub fn connection(&self) -> NeuronConnectionType {
        self.connection
    }

    pub fn mutated(&self) -> Gene {
        let bits = self.to_bits();
        let mut rng = rand::thread_rng();
        let position = rng.gen_range(0..bits.len());
        let mut new_bits = bits.clone();
        new_bits.set(position, !bits[position]);
        Gene::from_bits(&new_bits)
    }

    fn to_bits(&self) -> BitVec {
        let mut bits = BitVec::from_elem(16 + 8 + 8 + 2, false);
        let mut count = 0;
        for i in 0..16 {
            bits.set(count, (self.weight >> i) & 1 == 1);
            count += 1;
        }
        for i in 0..8 {
            bits.set(count, (self.input_number >> i) & 1 == 1);
            count += 1;
        }
        for i in 0..8 {
            bits.set(count, (self.output_number >> i) & 1 == 1);
            count += 1;
        }
        for i in 0..2 {
            bits.set(count, (self.connection as u8 >> i) & 1 == 1);
            count += 1;
        }
        bits
    }

    fn from_bits(bits: &BitVec) -> Gene {
        let mut count = 0;
        let mut gene = Gene::default();

        for i in 0..16 {
            gene.weight |= (bits[count] as u16) << i;
            count += 1;
        }

        for i in 0..8 {
            gene.input_number |= (bits[count] as u8) << i;
            count += 1;
        }

        for i in 0..8 {
            gene.output_number |= (bits[count] as u8) << i;
            count += 1;
        }

        let mut num = 0;
        for i in 0..2 {
            num |= (bits[count] as u8) << i;
            count += 1;
        }
        gene.connection = NeuronConnectionType::from_u8(num).unwrap();

        gene
    }
}

impl Default for Gene {
    fn default() -> Self {
        Gene {
            weight: 0,
            input_number: 0,
            output_number: 0,
            connection: NeuronConnectionType::InternalInternal,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Genome {
    genes: Vec<Gene>,
}

impl Genome {
    fn new(genes: Vec<Gene>) -> Genome {
        Genome { genes }
    }

    fn random(size: usize) -> Genome {
        let mut genes = Vec::with_capacity(size);
        for _ in 0..size {
            genes.push(Gene::random());
        }
        Genome::new(genes)
    }

    fn size(&self) -> usize {
        self.genes.len()
    }
}

mod test {
    #[test]
    fn bits_to_from() {
        use super::Gene;
        use super::NeuronConnectionType;
        use bit_vec::BitVec;

        fn test_bits(gene: Gene) {
            let bits = gene.to_bits();
            let gene2 = Gene::from_bits(&bits);
            assert_eq!(gene, gene2);
        }

        let gene = Gene::new(0x1234, 0x5, 0x6, NeuronConnectionType::InputOutput);
        let gene2 = Gene::new(666, 123, 44, NeuronConnectionType::InternalInternal);
        test_bits(gene);
        test_bits(gene2);
    }
}
