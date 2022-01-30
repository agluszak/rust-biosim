use crate::neural_network;
use crate::neural_network::Input;
use bevy::reflect::erased_serde::serialize_trait_object;
use bit_vec::BitVec;
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use rand::distributions::{Distribution, Standard};
use rand::Rng;
use variant_count::VariantCount;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Gene {
    weight: i16,
    input_index: u8,
    output_index: u8,
    from_internal: bool,
    to_internal: bool,
}

impl Distribution<Gene> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Gene {
        Gene::new(rng.gen(), rng.gen(), rng.gen(), rng.gen(), rng.gen())
    }
}

impl Gene {
    fn new(
        weight: i16,
        input_index: u8,
        output_index: u8,
        from_internal: bool,
        to_internal: bool,
    ) -> Gene {
        Gene {
            weight,
            input_index,
            output_index,
            from_internal,
            to_internal,
        }
    }

    pub fn random() -> Gene {
        rand::thread_rng().gen()
    }

    pub fn weight(&self) -> i16 {
        self.weight
    }

    pub fn input_index(&self) -> usize {
        self.input_index as usize
    }

    pub fn output_index(&self) -> usize {
        self.output_index as usize
    }

    pub fn is_from_internal(&self) -> bool {
        self.from_internal
    }

    pub fn is_to_internal(&self) -> bool {
        self.to_internal
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
        let mut bits = BitVec::from_elem(16 + 8 + 8 + 1 + 1, false);
        let mut count = 0;
        for i in 0..16 {
            bits.set(count, (self.weight >> i) & 1 == 1);
            count += 1;
        }
        for i in 0..8 {
            bits.set(count, (self.input_index >> i) & 1 == 1);
            count += 1;
        }
        for i in 0..8 {
            bits.set(count, (self.output_index >> i) & 1 == 1);
            count += 1;
        }
        bits.set(count, self.from_internal);
        count += 1;
        bits.set(count, self.to_internal);
        count += 1;
        bits
    }

    fn from_bits(bits: &BitVec) -> Gene {
        let mut count = 0;
        let mut gene = Gene::default();

        for i in 0..16 {
            gene.weight |= (bits[count] as i16) << i;
            count += 1;
        }

        for i in 0..8 {
            gene.input_index |= (bits[count] as u8) << i;
            count += 1;
        }

        for i in 0..8 {
            gene.output_index |= (bits[count] as u8) << i;
            count += 1;
        }

        gene.from_internal = bits[count];
        count += 1;
        gene.to_internal = bits[count];
        count += 1;

        gene
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Genome {
    genes: Vec<Gene>,
}

impl Genome {
    pub fn new(genes: Vec<Gene>) -> Genome {
        Genome { genes }
    }

    pub fn random(size: usize) -> Genome {
        let mut genes = Vec::with_capacity(size);
        for _ in 0..size {
            genes.push(Gene::random());
        }
        Genome::new(genes)
    }

    pub fn size(&self) -> usize {
        self.genes.len()
    }

    pub fn genes(&self) -> impl Iterator<Item = &Gene> {
        self.genes.iter()
    }
}

mod test {
    #[test]
    fn bits_to_from() {
        use super::Gene;
        use bit_vec::BitVec;

        fn test_bits(gene: Gene) {
            let bits = gene.to_bits();
            let from_bits = Gene::from_bits(&bits);
            assert_eq!(gene, from_bits);
        }

        let gene = Gene::new(0x1234, 0x5, 0x6, true, false);
        let gene2 = Gene::new(666, 123, 44, false, false);
        test_bits(gene);
        test_bits(gene2);
    }
}
