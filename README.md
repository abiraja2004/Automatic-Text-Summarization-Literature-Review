# Automatic-Text-Summarization-Literature-Review
Literature review of Automatic Text Summarization.
## 1. Introduction
Summaries are produced from single or multiple documents. Single document summarization produces an abstract, outline, or headline. Multiple document summarizations produce a gist of the content of the document-cluster such as news stories on the same event, or a set of web pages on the same topic.  

Summarization types are generic and query-focused/topic-constrained. Generic, summarizes the content of document-cluster. Query-focused and topic-constrained, summarize content based on user preference and the topic respectively. Email threads use topic-constrained because the topic represents it better than its content. Topic-constrained is essentially query-focused, where the topic is the query.  

Summarization methods are abstractive and extractive. Extractive methods emphasize content through extraction of sentences. Extraction identifies important sections of text and produces them verbatim (Das & Martins, 2007), (Sizov, 2010), and (Nenkova & McKeown, 2012). These systems work in general domains and use Information Retrieval techniques to produce superior content with little or no focus on form. Abstractive methods emphasize form, and aim to produce a grammatical and coherent summary, which usually requires advanced language generation techniques. These systems work in strict domains (e.g. news reports on terrorism).   

This document is about multi-document extractive summarization.

## 2. Related work
Organization of this section is as follows: The first subsection outlines the steps used in summarization. The second subsection introduces the terminology used throughout this document. Subsequent subsections describe four extractive summarization systems. The first, Maximal Marginal Relevance (MMR), is a research paper that pioneered relevance versus novelty. The second system, MEAD, pioneered cluster centroids. The last two systems, Clustering Linguistics AND Statistics Summarization Yield (CLASSY) and SumBasic plus its derivatives, are top scoring in the worldwide TAC (Text Analysis Conference) competition organized by the National Institute of Standards and Technology (NIST).
### 2.1. Steps in multi-document summarization
Multi-document summarization involves multiple sources of information that overlap, contradict, and supplement each other. Following are loosely the steps to select relevant and non-redundant sentences to produce a coherent and complete summary.
##### Segment Sentences
Sentence Segmentation extracts sentences from documents.
##### Simplify Sentences
Sentence simplification trims the redundant parts resulting in shorter/compressed grammatically correct sentences with content equivalent to their larger counterparts. Approaches vary regarding compression methods: some remove appositives and relative clauses from sentences (Blair-Goldensohn, et al., 2004), others develop patterns (Vanderwende, Suzuki, & Brockett, 2006) and compression rules through insights obtained by observing constructions in human-written summaries and lead sentences in stories (Zajic, Dorr, Lin, & Schwartz, 2007). Applying these rules and patterns to parse trees produce multiple compressed candidates for sentences (Toutanova, Brockett, Gamon, Jagarlamudi, Suzuki, & Vanderwende, 2007). Sentence selection algorithms then determine which compressed candidates to pick for the summary. 
##### Generate signature ngrams
Signature ngrams (excluding the stop-words) are salient/informative words and phrases in a document, document-cluster, and query. They can be identified by their frequency of occurrence (Luhn, 1958), by using title/heading as an important clue (Edmundson, 1969), by sentence position (Baxendale, 1958), and by log-likelihood ratios between sets of relevant and non-relevant documents (Dunning, March 1993 ), (Lin & Hovy, 2000). In general, stem the query and signature ngrams in order to identify other related ngrams.
##### Score Sentences
Score sentences through a formula that uses signature ngrams. 
##### Select Sentences
Select relevant and novel sentences through similarity measures between sentences. Some approaches identify novelty through clustering of sentences into themes and selecting one sentence from each theme (McKeown, Klavans, Hatzivassiloglou, Barzilay, & Eskin, 1999) or generating a composite sentence from each theme (Barzilay, McKeown, & Elhadad, 1999).
##### Order Sentences 
Order sentences to present a cohesive and coherent discourse. Some approaches include combining chronological ordering of events and topical relatedness of sentences (Barzilay, Elhadad, & McKeown, 2002), or using the Traveling Salesperson algorithm (Althaus, Karamanis, & Koller, 2004).
### 2.2. Terminology, Symbols, and Notations
Sc is a set of sentences in a cluster of documents, Sc = {sc1, sc2, …}, sc Є Sc      	       
Ss is a set of sentences in the Summary that are selected from Sc, so Ss = {ss1, ss2, …}, ss Є Ss   
(Sc – Ss) is a set of sentences in Sc minus those in Ss, and sc-s Є (Sc – Ss)   

Gc is a set of cluster-signature ngrams representing the cluster documents, Gc = {gc1, gc2, …}, gc Є Gc   
Gd is a set of document-signature ngrams representing a document in the cluster, Gd = {gd1, gd2, …}, Use Gdk for a specific kth document in the cluster      
Gq is a set of query-signature ngrams representing the user query, Gq = {gq1, gq2,…}    

A sentence sc is a set of cluster-signature ngrams selected from Gc, so sc = {gc2, gc7, …..}    
|sc| is the number of cluster-signature ngrams in sentence sc    
|gc| is the value of cluster-signature ngram calculated through measures such as TF-IDF, Term Frequency, log-likelihood ratios, Latent Dirichlet Allocation (LDA), etc.
### 2.3. Maximal Marginal Relevance (MMR)
MMR summarizes documents based on query-signature (Carbonell & Goldstein, 1998).
#### 2.3.1. Score Sentences
Score each sentence, s, in the cluster by its relevancy score, ωR(s), and its redundancy score, ωN(s). Determine the relevancy score, as a similarity measure (e.g. Cosine) of s with Gq. Determine the redundancy score, , as the similarity measure of s with the maximally redundant summary sentence ss. 
#### 2.3.2. Select Sentences
Select the maximum marginal relevant sentence that provides the best tradeoff between relevancy and redundancy. The θ parameter controls the tradeoff between relevance and redundancy, and its value is between 0 and 1. Through experiments, the authors suggest a reasonable strategy to select sentences as follows: Start with understanding the novelty of sentences in the region of the query by using a small θ (e.g. θ = 0.3). Then, use a reformulated query via relevance feedback with a large θ (e.g. θ = 0.7) to pick the important sentences.
Execution of the above equation results in moving sMMR from set (Sc – Ss) to set Ss. Executing the equation iteratively continues the moving process until a condition (e.g., minimum relevance threshold is attained, sufficient number of summary sentences is selected) is met that stops the iteration.
### 2.4. MEAD
MEAD summarizes documents based on cluster-signatures (Radev, Jing, Styś, & Tam, 2004).
#### 2.4.1. Generate cluster signature
Group similar documents into same clusters (Radev, Hatzivassiloglou, & McKeown, 1999) based on their TF-IDF values as follows: 
1. Represent each document with TF-IDF values of its document-signature unigrams. 
1. Represent a centroid of a cluster with TF-IDF values of its cluster-signature unigrams; calculate TF-IDF values as the weighted average of TF-IDF values of document-signature unigrams. 
1. Group a new document with a cluster whose centroid is close to TF-IDF values of the document-signature unigrams.
#### 2.4.2. Score Sentences
Score each sentence, , in a cluster based on its relevancy score, ωR(si), penalized by its redundancy score, ωN(si). Determine the relevancy score, , as a sum of three features - Centroid score ωg(si), Positional score ωp(si), and the First-sentence overlap score ωf(si). Base the Positional and the First-sentence overlap scores on the assumption that the first sentence in news articles is most relevant. The parameters – θg, θp, θf, - are given equal weights because a learning algorithm is not incorporated. 
The centroid score, , is the sum of the TF-IDF values of the cluster-signature unigrams in si. It determines the similarity of si with the cluster-signature.
The positional score, , assumes that the relevancy of sentences decreases with their distance from the first sentence in the document. The first sentence gets the highest score, equal to the highest centroid score, max[ωg(si)]. Successive sentences receive decreasing scores, and the last sentence (n) in the document has the lowest score of max[ωg(si)]/n. 
The first-sentence-overlap score, , assumes that the first sentence in the document is most relevant. This score is the inner product of vectors si and s1. Sentences that are more similar to the first sentence receive higher scores.
Determine the redundancy score, , as the product of the redundancy value, sim(si, so), penalized by the value of the highest scoring relevant sentence, max[ωR(si)]. The similarity measure, , includes stop-words and measures the similarity of si with the most relevant and redundant so. Note that sim(si, so) = 1 for identical sentences, and sim(si, so) = 0 for completely novel sentences. 
#### 2.4.3.Select Sentences
Select sentences, , similarly to the MMR procedure in Section 2.2.2.
#### 2.4.4.Order Sentences
Order sentences chronologically by their position within the document and by the dates of the documents.
### 2.5.CLASSY
CLASSY summarizes documents based on query-signatures (Conroy, Schlesinger, Kubina, Rankel, & O’Leary, 2011).
#### 2.5.1.Sentence Simplification
Simplify sentences by removing gerund phrases, relative clause attributives, attributions, many adverbs and all conjunctions, and ages (Conroy, Schlesinger, O’leary, & Goldstein, 2006).
#### 2.5.2.Generate query- and cluster-signature
Generate cluster-signature by using log-likelihood ratios with a large background corpus and with stop-words included (Conroy, Schlesinger, & O’Leary, 2007). Generate query-signature by extracting unigrams from topic description, and expanding those unigrams by populating all aspects (e.g. when, when, where) of a category (e.g. accident) using Google search, dictionaries and thesauruses (Conroy, Schlesinger, Rankel, & O’Leary, 2010).
#### 2.5.3.Score sentences
Score each sentence, , as an average of the sum of the values of its signature unigrams. The signature unigram value, , is the sum of the query, cluster, and maximum likelihood estimate values. The parameters consist of empirical values based on importance of gq, gc, and gρ. If g is a query-signature then |gq|= 1, otherwise 0. If g is a cluster-signature then |gc|= 1, otherwise 0. The value of gρ, , is the maximum likelihood estimate of the probability that g occurs in the summary; it is calculated from the signature unigram-sentence incidence matrix “A” consisting of “k” top scoring sentences corresponding to columns s1, s2,…,sk.
#### 2.5.4.Select Sentences
Select most relevant and novel sentences through non-negative (or L1-norm) QR factorization (Conroy & O’Leary, 2001). For update summaries, project the signature unigram-sentence matrix to minimize repetition of information in the base summary.
#### 2.5.5.Order Sentences
 Order selected sentences using the Traveling Salesperson algorithm (Conroy, Schlesinger, O’leary, & Goldstein, 2006), where unigram-overlap measures sentence similarity, and a Monte-Carlo method approximates the solution of this NP-hard problem.
### 2.6.SumBasic and SumFocus
Empirical studies validate that high frequency non-stop ngrams in documents are more likely to appear in human and in top ranking automatic summaries (Nenkova & Vanderwende, 2005). SumBasic exclusively exploits cluster unigram frequencies to generate generic summaries. SumFocus extends SumBasic to include query-focused summarization (Vanderwende L. , Suzuki, Brockett, & Nenkova, 2007) for task-focused summaries where the topic unigrams are highly predictive of the content of the summary.
#### 2.6.1.Generate query- and cluster-signatures
The signature unigrams are content ngrams, “g”. The stop-list consists of function words, pronouns, etc. The signature unigrams for SumBasic and SumFocus come from the cluster and from the cluster/query respectively. For SumFocus, the probability that a signature unigram appears in the summary, is the combination of probabilities derived from the query- and the cluster-signature unigrams. The θ parameter controls the tradeoff between assigning a higher weight to query or to cluster probability; its value is between 0 and 1. Empirically, θ = 0.9 because query-signature unigrams have a much higher likelihood of being in summaries. 
The probability derived from the query, is the number of times g occurs in the query divided by the total number of signature unigrams in the query. Smoothing assigns a small probability to signature unigrams that do not appear in the query. The probability derived from the cluster, is the number of times g occurs in the cluster divided by the total number of signature unigrams in the cluster.
#### 2.6.2.Score Sentences
Score each sentence as an average probability of its signature unigrams.
#### 2.6.3.Select Sentences
Select sentences as follows:
1.Calculate ω(s) for all sentences.
2.Select a sentence with the highest ω(s).
3.Decrease p(g) of unigrams whose sentence is selected: new[p(g)]= old[p(g)]. old[p(g)]. Reducing the probability of selecting unigrams multiple times enables novelty in the selection process.
4.Go to Step 1 until a condition (e.g., sufficient number of summary sentences is selected) is met.
2.6.4.Order Sentences
There is no consideration for sentence ordering or cohesion. Ordering is a result of the sentence selection process. 
2.7.LDA based summarization systems
SumBasic and SumFocus have two major drawbacks. (1) They favor repetition of high frequency signature ngrams in the summary, which reduces the possibility of picking moderately frequent signature ngrams. (2) They do not distinguish between document-specific ngram frequencies concentrated in a document versus the cluster-specific ngram frequencies spread across documents. The Kullback-Lieber (KL) divergence and the LDA-like topic model minimize the former and the latter drawbacks respectively. The KL divergence, , between the target, pT(g), and the approximating, pS(g), distributions is zero when they are identical and strictly positive otherwise. Following are summarization approaches/systems that use LDA or an LDA-derivative to generate cluster- and query-signatures.
2.7.1.TopicSum for generic summaries
TopicSum (Topic Summarization) is a generic summarizer (Haghighi & Vanderwende, 2009). The LDA model represents documents as mixtures of latent topics, where a topic is a probability distribution over words. It learns the following three topics: Background, Document-Specific, and Cluster-Specific. The background probability distribution consists of stop-words and other non-content ngrams that appear among all documents. The Document-Specific distribution consists of content ngrams in the document that does not appear in other documents. The Cluster-Specific distribution consists of content ngrams spread across the documents in the cluster. Each sentence is a distribution of three topics. Each unigram position of the sentence has a topic, and a unigram from the topic distribution. TopicSum greedily adds sentences to the summary as long as they decrease KL-divergence between the cluster-specific (target) and the summary (approximate) distributions.
2.7.2.BayesSum for query-focused summaries
BayeSum (Bayesian summarization) is a query-focused summarizer (Daumé III & Marcu, 2006). It requires a set of known relevant documents for a given query. A document consists of a mixture of three components: a general English component, a query-specific component, and a document-specific component. Sentences have a continuous probability distribution from each component. Words in a sentence have one of the three components. BayeSum greedily adds sentences to the summary using a strategy similar to MMR (Daumé III & Marcu, 2005).
2.7.3.DualSum for Update Summaries
DualSum modifies TopicSum to generate update summaries of cluster B of recent documents that has information not found in cluster A of earlier documents (Delort & Alfonseca, 2012, April). The LDA model learns the following four topics: Background, Document-Specific, Joint, and Update. The Joint distribution has information common to A and B. The Update distribution has information in B that is not present in A; it is zero for cluster A. 
The target distribution is a mixture of Update and Joint distributions. Empirical results show that the Update distribution alone produces lower results because it disregards generic ngrams about the topic described. A Joint distribution alone produces good results because it includes ngrams from both clusters. A mixture weight of 0.7 for Joint and 0.3 for Update produces the best results. DualSum generates summaries by greedily picking sentences and using KL-divergence. 
3.Papers and brief write-up on them
Wang et. al. (2009) propose a Bayesian approach for summarization that does not use KL for re-ranking. In their model, Bayesian Sentence based Topic Models, every sentence in a document is assumed to be associated to a unique latent topic. Once the model parameters have been calculated, a summary is generated by choosing the sentence with the highest probability for each topic.
Wallach, H. M. (2006, June). Topic modeling: beyond bag-of-words. In Proceedings of the 23rd international conference on Machine learning (pp. 977-984). ACM.  This article explores a hierarchical generative probabilistic model that incorporates both n-gram statistics and latent topic variables by extending a unigram topic model to include properties of a hierarchical Dirichlet bigram language model.
Louis, A., & Nenkova, A. (2013). Automatically assessing machine summary content without a gold standard. Computational Linguistics, 39(2), 267-300. http://www.newdesign.aclweb.org/anthology/J/J13/J13-2002.pdf  
This article discusses metrics for summary evaluation when human summaries are not present. On average, the JS divergence measure is highly predictive of summary quality, so it is better than KL divergence. Input–summary similarity based only on word distribution works well for evaluating summaries of cohesive-type inputs. 

Rankel, P. A., Conroy, J. M., Dang, H. T., & Nenkova, A. (2013). A Decade of Automatic Content Evaluation of News Summaries: Reassessing the State of the Art.    How good are automatic content metrics for news summary evaluation?
Barzilay, R., & Lapata, M. (2008). Modeling local coherence: An entity-based approach. Computational Linguistics, 34(1), 1-34. http://acl.ldc.upenn.edu/P/P05/P05-1018.pdf 
Misra, H., Cappé, O., & Yvon, F. (2008, August). Using LDA to detect semantically incoherent documents. In Proceedings of the Twelfth Conference on Computational Natural Language Learning (pp. 41-48). Association for Computational Linguistics. http://perso.telecom-paristech.fr/~cappe/papers/08-conll.pdf 
Based on the premise that a true document contains only a few topics and a false document is made up of many topics, it is asserted that the entropy of the topic distribution will be lower for a true document than that for a false document.

Jordan Boyd-Graber, David Blei, and Xiaojin Zhu. A topic model for word sense disambiguation. In Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning (EMNLP-CoNLL ’07), pages 1024–1033, 2007. http://aclweb.org/anthology//W/W13/W13-0102.pdf
This paper develops LDA with WORDNET (LDAWN), an unsupervised probabilistic topic model that includes word sense as a hidden variable. It develops a probabilistic posterior inference algorithm for simultaneously disambiguating a corpus and learning the domains in which to consider each word.

Haghighi, A., & Vanderwende, L. (2009, May). Exploring content models for multi-document summarization. In Proceedings of Human Language Technologies: The 2009 Annual Conference of the North American Chapter of the Association for Computational Linguistics (pp. 362-370). Association for Computational Linguistics. http://aclweb.org/anthology//N/N09/N09-1041.pdf
This paper extends SumBasic incrementally into KLSum, TopicSum, and HierSum, thereby increasing the Rouge scores along the way.

Daumé III, H., & Marcu, D. (2006, July). Bayesian query-focused summarization. In Proceedings of the 21st International Conference on Computational Linguistics and the 44th annual meeting of the Association for Computational Linguistics (pp. 305-312). Association for Computational Linguistics. http://acl.ldc.upenn.edu/P/P06/P06-1039.pdf 
Query focused model that is supposedly similar to (Haghighi & Vanderwende, 2009). Provides terminology, such as pd(.), and an absolutely necessary background information to understand (Haghighi & Vanderwende, 2009). 

Christensen, J., Mausam, S. S., & Etzioni, O. (2013). Towards Coherent Multi-Document Summarization. In Proceedings of NAACL-HLT (pp. 1163-1173). http://www.aclweb.org/anthology/N/N13/N13-1136.pdf 
This paper introduces, G-Flow, a joint model for selection and ordering that balances coherence and salience. G-Flow’s core representation is a graph that approximates the discourse relations across sentences based on indicators including discourse cues, deverbal nouns, co-reference, and more. This graph enables G-FLOW to estimate the coherence of a candidate summary. Additionally, while incorporating coherence into selection, their tapproach also performs joint selection and ordering.

Delort, J. Y., & Alfonseca, E. (2012, April). DualSum: a Topic-Model based approach for update summarization. In Proceedings of the 13th Conference of the European Chapter of the Association for Computational Linguistics (pp. 214-223). Association for Computational Linguistics. http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.230.9321&rep=rep1&type=pdf#page=238 
This paper modifies TopicSum to generate Update Summaries.

Mason, R., & Charniak, E. (2011, June). Extractive multi-document summaries should explicitly not contain document-specific content. In Proceedings of the Workshop on Automatic Summarization for Different Genres, Media, and Languages (pp. 49-54). Association for Computational Linguistics. http://aclweb.org/anthology//W/W11/W11-05.pdf#page=61
This paper modifies the existing system, HIERSUM (see Haghighi & Vanderwende, 2009 above), so sentences are penalized for containing content that is speciﬁc to the documents they were extracted from. It outperforms the original HIERSUM in pairwise user evaluation.

Mason, R., & Charniak, E. BLLIP at TAC 2011: A General Summarization System for a Guided Summarization Task. http://www.nist.gov/tac/publications/2011/participant.papers/BLLIP.proceedings.pdf  This is the same paper above that was at TAC 2011. The system’s summaries are relatively weak in linguistic quality, and have more redundancies than the average submitted summaries. For future work, the summaries could be improved by compressing the extracted sentences to remove redundant information, and reordering the sentences in order to make the summaries easier to read.

Darling, W. M., & Song, F. (2011). Pathsum: A summarization framework based on hierarchical topics. on Automatic Text Summarization 2011, 5. http://www.cs.toronto.edu/~akennedy/publications/proceedings_ts11.pdf#page=9 PathSum, is a high-performing hierarchical-topic based single- and multi-document automatic text summarization framework. This approach leverages Bayesian nonparametric methods to model sentences as paths through a tree and create a hierarchy of topics from the input in an unsupervised setting.

Blei, D. M., Griffiths, T. L., & Jordan, M. I. (2010). The nested chinese restaurant process and bayesian nonparametric inference of topic hierarchies. Journal of the ACM (JACM), 57(2), 7. http://www.cs.princeton.edu/~blei/papers/BleiGriffithsJordan2009.pdf  
This paper presents the nested Chinese restaurant process (nCRP), a stochastic process which assigns probability distributions to infinitely-deep, infinitely-branching trees. The previous paper “Darling, W. M., & Song, F. (2011). Pathsum: ….” uses the Chinese Restaurant Process to build hLDA.

Aletras, N., & Stevenson, M. (2013). Evaluating topic coherence using distributional semantics. In Proceedings of the 10th International Conference on Computational Semantics (IWCS’13)–Long Papers (pp. 13-22).   
This paper introduces distributional semantic similarity methods for automatically measuring the coherence of a set of words generated by a topic model. Newman et al. (2010b) - It is assumed that a topic is coherent if all or most of its words are related. Results show that word relatedness is better predicted using the distribution-based Pointwise Mutual Information (PMI) of words rather than knowledge-based measures.

Alfonseca, E., Pighin, D., & Garrido, G. HEADY: News headline abstraction through event pattern clustering. http://static.googleusercontent.com/external_content/untrusted_dlcp/research.google.com/en/us/pubs/archive/41185.pdf
This paper presents HEADY, an abstractive headline generation system based on the generalization of syntactic patterns by means of a Noisy-OR Bayesian network.

Lau, J. H., Baldwin, T., & Newman, D. (2013). On collocations and topic models. ACM Transactions on Speech and Language Processing (TSLP), 10(3), 10. http://ww2.cs.mu.oz.au/~tim/pubs/tslp2013-topiccoll.pdf
This paper incorporates bigrams in the document representation, and improves topic coherence.

Campr, M., & Jezek, K. Comparative Summarization via Latent Dirichlet Allocation. http://ceur-ws.org/Vol-971/poster11.pdf
This paper has a good description of LDA and LSA. It does Update Summarization. It aims to explore the possibility of using Latent Dirichlet Allocation (LDA) for multi-document comparative summarization which detects the main differences in documents. The idea is to use the LDA topic model to represent the documents, compare these topics and select the most significant sentences from the most diverse topics, to form a summary. To compare two vectors to gain the best results, try two possibilities: cosine similarity and Pearson correlation. From these two options, cosine similarity gives better results and comes out as a better choice, even if the precision is only higher in the order of tenths percent.

Celikyilmaz, A., & Hakkani-Tur, D. (2010, July). A hybrid hierarchical model for multi-document summarization. In Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics (pp. 815-824). Association for Computational Linguistics. 
This paper, formulates extractive summarization as a two step learning problem building a generative model for pattern discovery and a regression model for inference. It calculates scores for sentences in document clusters based on their latent characteristics using a hierarchical topic model. Then, using these scores, it trains a regression model based on the lexical and structural characteristics of the sentences, and use the model to score sentences of new documents to form a summary.

Celikyilmaz, A., Hakkani-Tur, D., & Tur, G. (2010, June). LDA based similarity modeling for question answering. In Proceedings of the NAACL HLT 2010 Workshop on Semantic Search (pp. 1-9). Association for Computational Linguistics.  
Given a topic z, the similarity between query and sentence is measured via transformed information radius (IR). It ﬁrst measure the divergence at each topic using IR based on Kullback-Liebler (KL) divergence. The divergence is transformed into similarity measure. To measure the similarity between probability distributions it opts for IR instead of commonly used KL because with IR there is no problem with inﬁnite values and it is also symmetric. In summary, sim1LDA is a measure of lexical similarity on topic-word level and sim2LDA is a measure of topical similarity on passage level. Together they form the degree of similarity DESLDA(s, q) = sim1LDA(q, s) x sim2LDA(q, s).
