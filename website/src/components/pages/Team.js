import React, { useEffect } from 'react';
import '../../App.css';

const current = [
  {
    name: 'Yuxuan Zhang',
    title: 'Data Scientist',
    image: `${process.env.PUBLIC_URL}/yuxuan.jpg`
  }
];

const contributors = [
  {
    name: 'Shruti Agarwal',
    image: `${process.env.PUBLIC_URL}/shruti.png`
  },
  {
    name: 'Yuluan Cao',
    image: `${process.env.PUBLIC_URL}/yuluan.jpg`
  },
  {
    name: 'Gina Chen',
    image: `${process.env.PUBLIC_URL}/gina.jfif`
  },
  {
    name: 'Priya DCosta',
    image: `${process.env.PUBLIC_URL}/priya.png`
  },
  {
    name: 'Nikhil Kumar',
    image: `${process.env.PUBLIC_URL}/nikhil.png`
  },
  {
    name: 'Pradnaya Pathak',
    image: `${process.env.PUBLIC_URL}/pradnaya.png`
  },
  {
    name: 'Evan Rowbotham',
    image: `${process.env.PUBLIC_URL}/evan.png`
  },
  {
    name: 'Yashveer Singh Sohi',
    image: `${process.env.PUBLIC_URL}/yashveer.png`
  },
  {
    name: 'Amy Zheng',
    image: `${process.env.PUBLIC_URL}/amy.png`
  },
  {
    name: 'Eric Zhong',
    image: `${process.env.PUBLIC_URL}/eric.jfif`
  },
  {
    name: 'Helena Zhou',
    image: `${process.env.PUBLIC_URL}/helena.png`
  }
];

function Team() {
  useEffect(() => {
    document.title = "Team | Team Communication Toolkit";
    window.scrollTo(0, 0)
  }, []);

  return (
    <div className='team-container'>
      <h1 className='team'>
        Meet Our Team
      </h1>

      <img src={`${process.env.PUBLIC_URL}/xinlan-emily-hu.jpg`} alt={'Xinlan Emily Hu'} className="emily-image" style={{ alignSelf: 'center' }} />
      <div className="emily">
        <div className='emily-member'>
          <h2> Xinlan Emily Hu </h2>
          <h4> Project Lead </h4>
          <h3> Postdoctoral Associate at MIT</h3>
        </div>
      </div>

      <h1 class="team-headers"> Current Members </h1>
      <div className="current">
        {current.map((member, index) => (
          <div key={index} className='current-member'>
            <img src={member.image} alt={member.name} className="current-image" />
            <h2>{member.name}</h2>
            <h3> {member.title} </h3>
          </div>
        ))}
      </div>

      <h1 class="team-headers"> Package Contributors </h1>
      <div className="alumni">
        {contributors.map((member, index) => (
          <div key={index} className='alumni-member'>
            <img src={member.image} alt={member.name} className="alumni-image" />
            <h2>{member.name}</h2>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Team;
