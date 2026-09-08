import React, { useEffect } from 'react';
import '../../App.css';

function Research() {
  useEffect(() => {
    document.title = "Research | Team Communication Toolkit";
    window.scrollTo(0, 0)
  }, []);

  return (
    <div className='research-container'>
      <h1 className='research'>
        Research
      </h1>
      <div className='cta-container'>
        <a
          className='cta-button'
          href='https://osf.io/preprints/psyarxiv/pz98q'
          target='_blank'
          rel='noopener noreferrer'
        >
          <span className='cta-label'>Read the Paper</span>
          <span className='cta-sublabel'>team_comm_tools: A Python Toolkit for Exploring Text Conversations in Groups</span>
        </a>
      </div>

      <h1 className='research-headers'> Abstract </h1>
      <p>
        Conversation is a rich window to our social world. Through it, humans coordinate their activities, resolve interpersonal disputes, and build lasting social bonds.
        However, the full potential of conversation data often goes unrealized, because analyzing a conversation requires making contingent and costly decisions about which
        constructs to study, how to measure them, and at what unit of analysis to do so. At the start of a project, however, an analyst may not know which constructs merit
        this investment. To lower the cost of a first-pass analysis, I introduce an omnibus tool for quantifying conversation: the Team Communication Toolkit
        (team_comm_tools on the Python Package Index; abbreviated TCT). The TCT is an open-source Python package that extracts 166 features at three units of analysis
        (the utterance, the speaker, and the conversation) from any corpus of text-based conversations. Requiring just four input columns and fewer than 10 lines of code,
        the TCT replaces what would otherwise be a patchwork of processing pipelines with a single streamlined interface. In a case study of 803 online public goods games,
        I use the TCT to explore which attributes of the talk preceding a decision predict how much a group contributes. I show that conversation is a meaningful signal of
        group outcomes only during a critical early-game period. 27 measures survive false-discovery-rate correction and replicate on held-out games. Together, they suggest
        a high-level interpretation that early-game conversations provide an honest signal of engagement, which is influential because social impressions are still nascent.
      </p>

      <h1 className='research-headers'> Applications </h1>
      <p> Examples of research questions that we are currently exploring with the toolkit include: </p>
      <div className='bullet-points'>
        <ul>
          <li> <strong> Analyzing Conflict: </strong> Identifying conversational markers that indicate whether or not a conflict is likely to be resolved productively;</li>
          <li> <strong> Analyzing Negotiations: </strong> Identifying conversational markers that suggest that parties are likely to feel satisfied after a negotiation; </li>
          <li> <strong> Analyzing Longitudinal Friendships</strong> Identifying conversational attributes of conversations among groups of college students whose text message chats remain active over many months, as opposed to fizzling out early. </li>
        </ul>
      </div>

      <br />
      <br />
      <p> <strong> We’re interested in working with you / hearing about YOUR ideas for studying conversations! </strong>
        We hope that other researchers will use the Team Conversation Toolkit to study research questions of their own.
      </p>

      <h2 className='research-subheaders'> Collaborations </h2>
      <p> If you are interested in a collaboration, please reach out to <a href='https://xinlanemilyhu.com'> Xinlan Emily Hu</a>.</p>


      <h2 className='research-subheaders'> Citation </h2>
      <p> If you use the Team Communication Toolkit in your work, please use the following citation: </p>
      <br />
      <div className='bullet-points'>
        <ul> Hu, Xinlan Emily. “team_comm_tools: A Python Toolkit for Exploring Text Conversations in Groups.” PsyArXiv, 2026. <a href='https://doi.org/10.31234/osf.io/pz98q'> https://doi.org/10.31234/osf.io/pz98q </a> </ul>
      </div>

    </div>
  );
}

export default Research;