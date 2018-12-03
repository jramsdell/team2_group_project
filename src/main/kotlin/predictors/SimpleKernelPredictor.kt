package predictors

import components.ComponentRepresentation
import components.NearestNeighborComponent
import components.StochasticComponent
import components.TrainingVectorComponent
import containers.EmailSparseVector
import edu.unh.cs753.predictors.LabelPredictor
import edu.unh.cs753.utils.SearchUtils
import org.apache.lucene.search.IndexSearcher


class SimpleKernelPredictor(searcher: IndexSearcher, val rep: ComponentRepresentation = ComponentRepresentation.FOURGRAM) : LabelPredictor(searcher) {
    val trainingComponent = TrainingVectorComponent(searcher, rep = rep)
    val embedded = trainingComponent.vectors.map { e -> trainingComponent.embed(e, trainingComponent.basisCollection[0]) }
    val stochastic = StochasticComponent(nBasis = embedded.first().components.size,  trainingVectors = embedded, holdout = embedded)
    val weights = when(rep) {
        ComponentRepresentation.UNIGRAM -> listOf(-0.11479406946883715, -0.016407518815816584, 0.2794013521076791, 0.0, -0.07568119037073924, -0.02955862675290243, -0.01905101303156586, -0.04071444824659089, 0.027242980684074127, -0.11701605044422102, -0.05126861319898766, 0.009337096418218133, -0.03689349208859195, 0.01612988401908545, -0.04964224701686341, 0.11283218712454564, -0.0674886362629083, 0.011065143742338794, 0.009253503990885552, -0.037743590810332156, -0.05109779717335123, 0.09210999067003103, -0.16507843848947962, 0.06756249239641134, 0.0, 0.0, -0.032298798336768776, 0.15592612759834826, 0.0, 0.08611572358337014, 0.04908056774801199, 0.008818118622463346, -0.11624665018714143, -0.11201435443781291, 0.21922004731083963, 0.18214601524891444, -0.015635206665072832, -0.021296500517695056, 0.04390593940666053, -0.035568023224744234, 0.007205846695394814, 0.0, 0.06820674947889084, -0.08013247599570637, 0.0498579277054766, 0.05324519279634768, -0.10632491058732482, -0.08405230148993073, 0.0, 0.008222241340649162, -0.03934978003850108, 0.05226454036054519, 0.005468739423301505, 0.04246678233784524, 0.018391067312884705, 0.06997092330943525, -0.10277128240337498, -0.00464077776926116, 0.2609409676356501, -0.070049108462229, 0.365910513322395, 0.3366058560134507, 0.0, 0.015466635037236113, -0.067087402824274, -0.019188915862640737, 0.08985676063585478, 0.3362607767355929, 0.04561947395000581, -0.03134893159355528, 0.015160490836969026, -0.059913085747335955, 0.34358301627252374, -0.027310180912654643, -0.1389163451236675, -0.026700979874207713, 0.0, 0.009337096418218133, 0.1441765801698335, -0.02309487775240268)
//        ComponentRepresentation.UNIGRAM -> listOf(-0.044602846654850856, 0.0, 0.28030504219618807, 0.0060927320794421334, -0.045564092673498355, -0.01772597537367964, 0.014780405683218804, -0.06904083747229907, -0.0018649770236209562, -0.19070748880979532, -0.047736781467086445, 0.0, -0.04774688846053389, 0.0, -0.07345750895379155, 0.15207386548579568, 0.011367744059552504, 0.034642877629406835, 0.00794589643083145, -0.032247452149279876, -0.029197243998135138, 0.15002235151730445, -0.16122714315488315, 0.06871430952003847, 0.06286719127165787, 0.0, 0.0, 0.06311323874749693, 0.0, 0.08937437552446853, 0.02512763903361914, -0.024459142235287733, -0.11521485834204001, -0.08177190085503533, 0.1297636215206445, 0.16446082438420445, -0.016313876742120453, -0.07124005311529939, 0.0, 0.0, 0.008942637569930582, 0.015485058226897289, 0.02775843299143473, 0.0, 0.011285373426487518, 0.023710601332794474, -0.055741960545324584, -0.0404896795503528, 0.038347527705260336, -0.025778730725535957, 0.006500820895975974, 0.04905937810062149, 0.009431997757327568, 0.0760304615010925, -0.0650964682283002, 0.03389471358146576, -0.0779127209304685, 0.00787544931081657, 0.27490789689720047, -0.06388865742974825, 0.34397553013805093, 0.29821395731380385, -0.011947372033963915, 0.0, -0.0319198249558476, -0.035836008162674175, 0.19612834102369045, 0.3433237006623398, -0.02451188217176543, -0.0632041562245756, 0.0, -0.07015539627557414, 0.42728644155543805, -0.07019260423340624, -0.13203369431162126, 0.007170225425260177, -0.042314033262481315, -0.006874801339880101, 0.08601753074359074, -0.042381889031719126)
        ComponentRepresentation.FOURGRAM -> listOf(-0.02721204371550404, -0.03781258757492228, 0.10239169603444673, 0.016091631331343698, -0.06444525174582899, -0.07773076019308929, 0.0, -0.043763243573262604, 0.0, -0.22891836336559016, 0.05966848227173798, 0.0, -0.06758458318236864, 0.0053212191069331284, 0.1878804435659837, 0.09875886185871513, 0.0, 0.04681992548478128, 0.0, -0.1490854502248965, 0.06759813198635502, -0.06890341098618002, 0.061821574307077236, 0.0, 0.18441826219896199, -0.0791588134915079, 0.07583785592673818, 0.06333044004627689, 0.07628963712969591, 0.0, 0.15581787766586683, -0.2907489838380649, 0.0, 0.14635065633400485, 0.1533405340475674, -0.026070047724736038, -0.08651047228863488, 0.0, 0.21556700896864903, -0.04411314158371539, 0.21739363885963703, 0.03218458987846369, 0.0, -0.020479038897958123, 0.0, -0.10576425949377362, 0.0, -0.15951574401295127, 0.12368504040614743, 0.01932720577501344, 0.0, -0.04493326903219356, 0.0, 0.0, 0.019500972139697374, 0.0, -0.050520350037937735, 0.35141097022681383, 0.22933949663997724, 0.03316324963022803, -0.019674719089207846, -0.2040382081548524, -0.08874347409112487, 0.012085664094014747, 0.3072401374347469, 0.1438870004861085, 0.0, 0.16452062697821346, -0.07954511795963949, 0.09503876578515745, -0.09289440457821606, -0.14581501020011772, 0.09529130325742813, 0.019946115746634553, 0.15633516220033888, -0.009783601884126317, -0.07377766519406176, 0.0620679824968646, -0.04121392821484373, 0.0)
//        ComponentRepresentation.FOURGRAM -> listOf(-0.052765906251957026, -0.02446180152589214, 0.13705195881253493, -0.019390225025338593, -0.060408293395592894, -0.07074077642289373, -0.010352634769678034, -0.04408950523708778, 0.0, -0.2054999982177734, -0.04181950039144662, 0.03316446549040038, -0.0659510422533571, 0.0, 0.15245988452417128, 0.19530896299829759, 0.0, 0.07052443323215155, 0.0, -0.09879622408302673, 0.06963727801430919, -0.07521971332996225, 0.05986507955755624, 0.012546895904107383, 0.10682147601977292, -0.11276477280237399, 0.08773743172083129, 0.048521261796812244, 0.04975754864290173, 0.024017609013943106, 0.1487735400855994, -0.18300677579579627, 0.0, 0.1142423604972257, 0.175162685740105, -0.02531976965802162, -0.23618932381044613, 0.04348495043034352, 0.22047812117887458, -0.0652375122311081, 0.1684554813264413, -0.026904631400267772, 0.1117803955385106, -0.03016418634288951, 0.0, -0.17314587820710542, 0.012625312549910613, -0.12865198219355264, 0.10231444609133801, -0.020192919299987497, 0.1206491691481195, 0.10161219248270578, 0.0, -0.025234410249688737, 0.023574944370465215, 0.0, -0.06424423831181118, 0.3933512855336812, 0.21493321021399409, 0.047088590920368295, 0.022326190467933466, -0.17958422400817362, -0.07813210860979297, 0.08419548942183618, 0.2746785381096195, 0.08474759262641852, -0.0504387643430551, 0.1624252760127814, -0.0711259088946251, 0.11602675262866191, -0.08319747832849093, -0.12352728190542411, 0.08897646813940593, 0.0, 0.10965974965838254, 0.0, -0.1336435071890605, 0.09880842828185922, 0.0, 0.0)
        ComponentRepresentation.BIGRAM -> listOf(-0.005352130465133843, 0.0, 0.24796617850038147, 0.006879309168350877, 0.0, 0.0, -7.353612657021296E-4, -0.009198228431567571, -0.00420213778412689, 0.16767544999577386, 0.11265397511265964, 0.00420213778412689, -0.00485800383809661, 0.00420213778412689, 0.08910166145468132, 0.018145767030461585, 0.08223102862882381, 0.0, 0.004233874580308161, -0.01431789524408061, 0.08007249072132594, -0.00859321348229521, 0.0045950806503809, 0.04591212042612502, 0.00608200673819061, -0.01439183906064045, 0.010382842451456981, 0.0, 0.0129442997632919, 0.00864676215042183, 0.06154669535192441, -0.023168963082126834, -0.024400017785564835, 0.2110050852719941, 0.018890934864929255, -0.0014085657560751403, -0.08138412162889219, 0.0, 0.11917786780224694, 0.0468545784620589, 0.02162403492516374, 0.010255239795468584, -0.045767710507495306, -0.009203797655007947, -0.020547614982486095, -0.07543318189675173, -0.02104365144307636, -0.005212607520150772, 0.014926242957886209, -0.0012901879616727548, 0.38468336795921715, 0.26446441154871053, 0.0, 0.0, 0.002282366990600821, 0.1826719627649862, -0.024771532967280192, 0.32013254266561275, 0.07821653706068206, 0.0, -0.006043377575237191, -0.018526865480426348, 0.0, 0.009558116113988437, 0.0, 0.005528686399113531, 0.0042259180333195975, 0.06283309527362893, -0.0036522207501151443, -0.6303646209178437, 0.011799146775883161, -0.1577205156023903, 0.008901484162729024, -0.0047272970313508335, 0.05584458698945422, 0.0, 0.0, 0.0, -0.004164120244652408, 0.00420213778412689)
    }
    var labeler = stochastic.myLabeler(weights)

    init {
        stochastic.memoizedHamDist = stochastic.createNormalDist(weights, stochastic.hamVectors)
        stochastic.memoizedSpamDist = stochastic.createNormalDist(weights, stochastic.spamVectors)
//        println(stochastic.memoizedHamDist.mean)
//        println(stochastic.memoizedSpamDist.mean)
        labeler = stochastic.myLabeler(weights)
    }

    override fun predict(tokens: MutableList<String>?): String {
        val dist = tokens!!
            .run {
                when (rep) {
                    ComponentRepresentation.UNIGRAM  -> this
                    ComponentRepresentation.FOURGRAM -> flatMap { trainingComponent.createCharacterGrams(it, 4) }
                    ComponentRepresentation.BIGRAM   -> trainingComponent.createBigrams(this)
                } }
            .groupingBy { it }
            .eachCount()
            .map { it.key to it.value.toDouble() }
            .toMap()

        val dist2 = tokens!!
            .run { trainingComponent.createBigrams(this) }
            .groupingBy { it }
            .eachCount()
            .map { it.key to it.value.toDouble() }
            .toMap()

        val v = EmailSparseVector("", components = dist, bigrams = dist2)
        val embedding = trainingComponent.embed(v, trainingComponent.basisCollection[0])
        return labeler(embedding)
    }
}

fun main(args: Array<String>) {
    val searcher = SearchUtils.createIndexSearcher("index")
    val predictor = SimpleKernelPredictor(searcher)
    predictor.evaluate()
}