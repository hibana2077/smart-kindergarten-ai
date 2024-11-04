'use client';

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Users, Plus, Pencil, School, LogOut } from 'lucide-react';

const TeacherProfile = () => {
  const teacherData = {
    name: "王小明",
    role: "主任教師",
    avatar: "WX" // 老師名字縮寫
  };

  return (
    <Card className="w-fit absolute top-6 right-6">
      <CardContent className="flex items-center gap-3 p-3">
        <div className="w-10 h-10 rounded-full bg-blue-100 flex items-center justify-center">
          <span className="text-blue-600 font-medium">{teacherData.avatar}</span>
        </div>
        <div className="flex flex-col">
          <span className="font-medium">{teacherData.name}</span>
          <span className="text-sm text-gray-500">{teacherData.role}</span>
        </div>
        <Button variant="ghost" size="icon" className="ml-2">
          <LogOut className="w-4 h-4 text-gray-500" />
        </Button>
      </CardContent>
    </Card>
  );
};

const classData: ClassInfo[] = [
  {
    id: 1,
    name: '向日葵班',
    description: '幼幼班',
    year: 2024,
    studentCount: 20,
    bgColor: 'bg-yellow-100',
    coverColor: 'bg-yellow-500',
    textColor: 'text-yellow-900'
  },
  {
    id: 2,
    name: '櫻花班',
    description: '幼幼班',
    year: 2024,
    studentCount: 18,
    bgColor: 'bg-pink-100',
    coverColor: 'bg-pink-500',
    textColor: 'text-pink-900'
  },
  {
    id: 3,
    name: '梅花班',
    description: '幼幼班',
    year: 2024,
    studentCount: 22,
    bgColor: 'bg-blue-100',
    coverColor: 'bg-blue-500',
    textColor: 'text-blue-900'
  },
  {
    id: 4,
    name: '百合班',
    description: '幼幼班',
    year: 2024,
    studentCount: 19,
    bgColor: 'bg-purple-100',
    coverColor: 'bg-purple-500',
    textColor: 'text-purple-900'
  },
  {
    id: 5,
    name: '玫瑰班',
    description: '幼幼班',
    year: 2024,
    studentCount: 21,
    bgColor: 'bg-red-100',
    coverColor: 'bg-red-500',
    textColor: 'text-red-900'
  },
  {
    id: 6,
    name: '薰衣草班',
    description: '幼幼班',
    year: 2024,
    studentCount: 17,
    bgColor: 'bg-indigo-100',
    coverColor: 'bg-indigo-500',
    textColor: 'text-indigo-900'
  },
  {
    id: 7,
    name: '紫羅蘭班',
    description: '幼幼班',
    year: 2024,
    studentCount: 23,
    bgColor: 'bg-green-100',
    coverColor: 'bg-green-500',
    textColor: 'text-green-900'
  }
];

interface ClassInfo {
  id: number;
  name: string;
  description: string;
  year: number;
  studentCount: number;
  bgColor: string;
  coverColor: string;
  textColor: string;
}

const ClassCard = ({ classInfo, onSelect }: { classInfo: ClassInfo; onSelect: (classInfo: ClassInfo) => void }) => {
  return (
    <Card 
      className={`cursor-pointer transition-all duration-200 ${classInfo.bgColor}`}
      onClick={() => onSelect(classInfo)}
    >
      <CardHeader>
        <div className="flex justify-between items-start">
          <div className="flex items-center gap-2">
            <div className={`w-8 h-8 rounded-full ${classInfo.coverColor} flex items-center justify-center`}>
              <School className={`w-5 h-5 ${classInfo.textColor}`} />
            </div>
            <div>
              <CardTitle className="text-lg">{classInfo.name}</CardTitle>
              <CardDescription>{classInfo.description}</CardDescription>
            </div>
          </div>
          <Badge variant="outline">{classInfo.year}年</Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-center gap-2">
          <Users className="w-4 h-4 text-gray-500" />
          <span className="text-sm text-gray-600">
            {classInfo.studentCount} 位學生
          </span>
        </div>
      </CardContent>
      <CardFooter className="bg-white/50">
        <Button 
          variant="default"  // 改為 default variant 來顯示顏色
          className={`w-full ${classInfo.textColor} ${classInfo.coverColor} rounded-2xl`}
          onClick={(e) => {
            e.stopPropagation();
            onSelect(classInfo);
          }}
        >
          <Pencil className="w-4 h-4 mr-2" />
          管理班級
        </Button>
      </CardFooter>
    </Card>
  );
};

const AddClassDialog = () => {
  const [formData, setFormData] = useState({
    name: '',
    year: new Date().getFullYear().toString(),
    description: '',
    studentCount: ''
  });

  interface FormData {
    name: string;
    year: string;
    description: string;
    studentCount: string;
  }

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>): void => {
    e.preventDefault();
    console.log('New class:', formData);
  };

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Card className="cursor-pointer border-dashed bg-gray-50/50 hover:bg-gray-50 transition-colors">
          <CardContent className="flex flex-col items-center justify-center h-full py-12"> {/* 調整高度和內邊距 */}
            <div className="w-12 h-12 rounded-full bg-blue-100 flex items-center justify-center mb-4"> {/* 增加底部邊距 */}
              <Plus className="w-6 h-6 text-blue-600" />
            </div>
            <p className="text-lg font-medium text-blue-600">新增班級</p>
          </CardContent>
        </Card>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>新增班級</DialogTitle>
          <DialogDescription>
            請填寫新班級的基本資訊
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="name">班級名稱</Label>
            <Input
              id="name"
              value={formData.name}
              onChange={(e) => setFormData({...formData, name: e.target.value})}
              placeholder="例如：向日葵班"
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="year">學年度</Label>
            <Input
              id="year"
              value={formData.year}
              onChange={(e) => setFormData({...formData, year: e.target.value})}
              placeholder="例如：2024"
              required
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="description">班級描述</Label>
            <Input
              id="description"
              value={formData.description}
              onChange={(e) => setFormData({...formData, description: e.target.value})}
              placeholder="例如：幼幼班"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="studentCount">學生人數</Label>
            <Input
              id="studentCount"
              type="number"
              value={formData.studentCount}
              onChange={(e) => setFormData({...formData, studentCount: e.target.value})}
              placeholder="請輸入人數"
              required
            />
          </div>
          <DialogFooter>
            <Button type="submit">確認新增</Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
};

const TeacherClassManagement = () => {
  const handleClassSelect = (classInfo: ClassInfo): void => {
    console.log('Selected class:', classInfo);
  };

  return (
    <div className="p-6 max-w-7xl mx-auto relative"> {/* 添加 relative 定位 */}
      <TeacherProfile />
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">班級管理</h1>
      </div>

      <div className="mt-12">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {classData.map((classInfo) => (
          <ClassCard 
            key={classInfo.id} 
            classInfo={classInfo}
            onSelect={handleClassSelect}
          />
        ))}
        <AddClassDialog />
      </div>
      </div>
    </div>
  );
};

export default TeacherClassManagement;